from fastapi import APIRouter

from app.services.live_monitor import get_snapshot, MONITORED_LOCATIONS

router = APIRouter()


@router.get("/live/weather")
async def live_weather():
    """Current cached conditions for all automatically-monitored Indian
    regions (Goa, Mumbai, Delhi, ...). Updated by a background task every
    `alert_poll_interval_seconds` — this endpoint just reads the cache, so
    it's cheap to poll from a dashboard."""
    snapshot = get_snapshot()
    return {
        "locations": [snapshot[name] for name, _, _ in MONITORED_LOCATIONS if name in snapshot],
        "monitored_count": len(MONITORED_LOCATIONS),
    }
