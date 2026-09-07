"""
Adapter around a real weather API (Open-Meteo, free/no-key) so the rest of
the app never talks to a vendor format directly. Swap `fetch_forecast` /
`geocode_location` internals to point at IMD, GFS/WRF outputs, or another
provider without touching callers.

Optimizations over a naive implementation:
- A single shared httpx.AsyncClient with connection pooling + HTTP keep-alive,
  instead of opening a new TCP/TLS connection per request.
- A small in-memory TTL cache in front of both geocoding and forecast calls,
  since forecasts don't meaningfully change second-to-second and the same
  locations get asked about repeatedly (e.g. "Pune" from many users).
"""
import time

import httpx

from app.core.config import settings


class WeatherServiceError(Exception):
    pass


# --- Shared client (connection pooling / keep-alive) -----------------------
# Created once at app startup (see main.py lifespan) and reused for every
# request instead of opening a fresh connection each time.
_client: httpx.AsyncClient | None = None


async def init_client() -> None:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=10,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        raise WeatherServiceError("HTTP client not initialized — app startup did not run.")
    return _client


# --- Simple TTL cache --------------------------------------------------------
# Geocoding results are effectively static -> cache for a day.
# Forecasts change over time but not second-to-second -> cache for 10 min.
_GEOCODE_TTL = 24 * 60 * 60
_FORECAST_TTL = 10 * 60
_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str):
    hit = _cache.get(key)
    if hit is None:
        return None
    expires_at, value = hit
    if time.time() > expires_at:
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: object, ttl: int) -> None:
    _cache[key] = (time.time() + ttl, value)


# --- Public API ---------------------------------------------------------

async def geocode_location(place_name: str) -> tuple[float, float, str]:
    """Resolve a free-text place to coordinates using India-first geocoding.

    The country filter is applied by the geocoding API itself, not by a
    hard-coded city list. This makes ambiguous names such as "Goa" resolve to
    India while still supporting any Indian city, district, town, or village.
    """
    normalized = " ".join(place_name.strip().split())
    if not normalized:
        raise WeatherServiceError("Location cannot be empty.")

    country_code = (settings.default_country_code or "").strip().upper()
    cache_key = f"geo:{country_code}:{normalized.lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    params = {
        "name": normalized,
        "count": 10,
        "language": "en",
        "format": "json",
    }
    if country_code:
        params["countryCode"] = country_code

    try:
        resp = await _get_client().get(settings.geocoding_api_base, params=params)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise WeatherServiceError(f"Geocoding service failed for '{normalized}': {exc}") from exc

    results = resp.json().get("results") or []
    if not results:
        raise WeatherServiceError(
            f"Could not resolve '{normalized}' in {country_code or 'the available locations'}."
        )

    # Results are already ranked by Open-Meteo. We only use the country as a
    # final safety check so an unrelated country can never slip through when
    # India-first mode is enabled.
    if country_code:
        country_results = [
            item for item in results
            if (item.get("country_code") or "").upper() == country_code
        ]
        if not country_results:
            raise WeatherServiceError(
                f"Could not resolve '{normalized}' in country {country_code}."
            )
        results = country_results

    def rank(item: dict) -> tuple[int, int, int, int]:
        feature = (item.get("feature_code") or "").upper()
        # Prefer populated places/admin areas, then higher population. The
        # original API ordering remains the final tie-breaker.
        feature_priority = 0 if feature.startswith(("PPL", "ADM")) else 1
        population = int(item.get("population") or 0)
        return feature_priority, -population, results.index(item), 0

    top = min(results, key=rank)
    name = top.get("name") or normalized
    admin1 = top.get("admin1")
    country = top.get("country")
    parts = [str(name)]
    if admin1 and str(admin1).lower() != str(name).lower():
        parts.append(str(admin1))
    if country and str(country).lower() != str(name).lower():
        parts.append(str(country))
    resolved_name = ", ".join(parts)

    result = (float(top["latitude"]), float(top["longitude"]), resolved_name)
    _cache_set(cache_key, result, _GEOCODE_TTL)
    return result


async def fetch_forecast(latitude: float, longitude: float, days: int = 7) -> dict:
    """Fetch current + daily forecast for a coordinate.

    Open-Meteo supports up to 16 forecast days. We cap the caller input at 16
    and default to 7 so normal forecast questions are useful out of the box.
    """
    days = max(1, min(int(days), 16))
    cache_key = f"fc:{round(latitude, 2)}:{round(longitude, 2)}:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    resp = await _get_client().get(
        settings.weather_api_base,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max",
            "forecast_days": days,
            "timezone": "auto",
        },
    )
    resp.raise_for_status()
    data = resp.json()
    _cache_set(cache_key, data, _FORECAST_TTL)
    return data


# WMO weather codes -> short human description
_WEATHER_CODE_MAP = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def describe_weather_code(code: int) -> str:
    return _WEATHER_CODE_MAP.get(code, "unknown conditions")


def is_severe(daily: dict, day_index: int = 0) -> bool:
    """Very simple severity heuristic for demo alerting — replace with
    real IMD/NDMA thresholds in production."""
    code = daily["weather_code"][day_index]
    wind = daily["wind_speed_10m_max"][day_index]
    precip = daily["precipitation_sum"][day_index]
    return code in (65, 75, 82, 95, 96, 99) or wind > 50 or precip > 100
