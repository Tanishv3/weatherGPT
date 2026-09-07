"""
Proactive live weather monitoring for a fixed set of Indian regions.

Runs as a background asyncio task (started in main.py's lifespan). On each
poll cycle it fetches the forecast for every monitored location, checks the
existing severity heuristic (weather_service.is_severe), and:
  - always updates the in-memory snapshot used by GET /live/weather
  - broadcasts a WebSocket alert only on the *transition* into severe
    conditions (not every poll), so connected clients get pushed a rain/
    storm alert automatically without asking, but don't get spammed.
"""
import asyncio
import logging
from datetime import datetime, timezone

from app.core.config import settings
from app.services import weather_service
from app.services.connection_manager import manager
from app.services.weather_service import WeatherServiceError, describe_weather_code, is_severe

logger = logging.getLogger("weathergpt.live_monitor")

# Major regions worth watching by default — spans the west coast monsoon
# belt (Goa, Mumbai), east coast cyclone belt (Chennai, Kolkata, Bhubaneswar),
# and interior/north for broad coverage. Extend freely.
MONITORED_LOCATIONS: list[tuple[str, float, float]] = [
    ("Goa", 15.4909, 73.8278),
    ("Mumbai", 19.0760, 72.8777),
    ("Pune", 18.5204, 73.8567),
    ("Delhi", 28.6139, 77.2090),
    ("Bengaluru", 12.9716, 77.5946),
    ("Chennai", 13.0827, 80.2707),
    ("Hyderabad", 17.3850, 78.4867),
    ("Kolkata", 22.5726, 88.3639),
    ("Bhubaneswar", 20.2961, 85.8245),
    ("Ahmedabad", 23.0225, 72.5714),
    ("Jaipur", 26.9124, 75.7873),
    ("Thiruvananthapuram", 8.5241, 76.9366),
    ("Guwahati", 26.1445, 91.7362),
    ("Lucknow", 26.8467, 80.9462),
]

# name -> {"severe": bool, "current": {...}, "daily": {...}, "updated_at": iso str}
_snapshot: dict[str, dict] = {}

# name -> was it severe as of the previous poll (for edge-triggered alerts)
_previous_severity: dict[str, bool] = {}

_task: asyncio.Task | None = None


def get_snapshot() -> dict[str, dict]:
    """Current cached conditions for every monitored location. Safe to call
    from request handlers — just reads the in-memory dict."""
    return _snapshot


async def _poll_once() -> None:
    for name, lat, lon in MONITORED_LOCATIONS:
        try:
            forecast = await weather_service.fetch_forecast(lat, lon)
        except WeatherServiceError as e:
            logger.warning("live monitor: failed to fetch %s: %s", name, e)
            continue
        except Exception as e:
            logger.warning("live monitor: unexpected error for %s: %s", name, e)
            continue

        current = forecast.get("current", {})
        daily = forecast.get("daily", {})
        severe = is_severe(daily, 0) if daily.get("weather_code") else False

        _snapshot[name] = {
            "location": name,
            "severe": severe,
            "current": current,
            "today": {
                "condition": describe_weather_code(daily["weather_code"][0]) if daily.get("weather_code") else None,
                "precipitation_mm": daily.get("precipitation_sum", [None])[0],
                "wind_kmh": daily.get("wind_speed_10m_max", [None])[0],
                "temp_min": daily.get("temperature_2m_min", [None])[0],
                "temp_max": daily.get("temperature_2m_max", [None])[0],
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        was_severe = _previous_severity.get(name, False)
        if severe and not was_severe:
            condition = describe_weather_code(daily["weather_code"][0])
            await manager.broadcast({
                "type": "alert",
                "location": name,
                "message": (
                    f"⚠️ Automated alert: {condition} developing in {name} — "
                    f"{daily['precipitation_sum'][0]} mm precipitation, "
                    f"wind up to {daily['wind_speed_10m_max'][0]} km/h expected today."
                ),
                "data": _snapshot[name],
            })
            logger.info("live monitor: broadcast alert for %s (%s)", name, condition)
        _previous_severity[name] = severe


async def _run_loop() -> None:
    # First poll immediately so the dashboard/snapshot isn't empty on startup.
    await _poll_once()
    while True:
        await asyncio.sleep(settings.alert_poll_interval_seconds)
        await _poll_once()


def start() -> None:
    global _task
    if _task is None:
        _task = asyncio.create_task(_run_loop())


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
        _task = None
