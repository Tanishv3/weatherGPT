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
import asyncio

import certifi
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
            timeout=httpx.Timeout(15.0),
            verify=certifi.where(),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            follow_redirects=True,
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
    """Resolve a free-text place name to (lat, lon, resolved_name). Cached."""
    cache_key = f"geo:{place_name.strip().lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        resp = await _get_client().get(
            settings.geocoding_api_base,
            params={"name": place_name, "count": 10, "language": "en", "format": "json", "countryCode": "IN"},
        )
        resp.raise_for_status()
        results = resp.json().get("results")
    except (httpx.HTTPError, ValueError) as exc:
        # ValueError also covers json.JSONDecodeError — a non-JSON response
        # (proxy/antivirus intercept page, captive portal, upstream error
        # page served with a 200) must never escape as an unhandled crash.
        raise WeatherServiceError(f"Geocoding request failed for '{place_name}': {exc}") from exc
    if not results:
        raise WeatherServiceError(f"Could not resolve location: {place_name}")

    # WeatherGPT is focused on India. Prefer an Indian result whenever the
    # user asks for a state/city name that has multiple matches worldwide.
    indian = [r for r in results if str(r.get("country_code", "")).upper() == "IN"]
    if not indian:
        raise WeatherServiceError(f"Could not find an Indian location: {place_name}")

    # Open-Meteo's geocoding search doesn't rank by "how important is this
    # place" — an obscure hamlet with a similar/partial name can outrank the
    # actual city (e.g. "Panaji Muwara, Gujarat" beating "Panaji, Goa" for a
    # query of "Panaji"). Score candidates instead of blindly taking the
    # first Indian result:
    #   1. exact name match (case-insensitive) always wins
    #   2. otherwise prefer a name that starts with the query
    #   3. tie-break on population (bigger places are far more likely to be
    #      what the user meant by a bare city/capital name)
    query = place_name.strip().lower()

    def _match_score(r: dict) -> tuple[bool, bool, int]:
        name = str(r.get("name", "")).strip().lower()
        return (name == query, name.startswith(query), r.get("population") or 0)

    top = max(indian, key=_match_score)
    resolved_name = f"{top['name']}, {top.get('admin1', '')}".strip(", ")
    result = (top["latitude"], top["longitude"], resolved_name)
    _cache_set(cache_key, result, _GEOCODE_TTL)
    return result


async def fetch_forecast(latitude: float, longitude: float, days: int = 3) -> dict:
    """Fetch current + daily forecast for a coordinate. Cached ~10 min,
    rounded to ~1km grid so nearby requests share a cache entry."""
    cache_key = f"fc:{round(latitude, 2)}:{round(longitude, 2)}:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max",
        "forecast_days": days,
        "timezone": "auto",
    }

    # Open-Meteo can occasionally return transient 502/503/504 responses.
    # Retry with exponential backoff instead of immediately failing the app.
    last_exc = None
    for attempt in range(3):
        try:
            resp = await _get_client().get(settings.weather_api_base, params=params)
            if resp.status_code in (429, 502, 503, 504):
                resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            _cache_set(cache_key, data, _FORECAST_TTL)
            return data
        except (httpx.HTTPError, ValueError) as exc:
            # ValueError also covers json.JSONDecodeError — same rationale
            # as geocode_location above: never let a bad response body
            # escape as an unhandled exception.
            last_exc = exc
            if attempt < 2:
                await asyncio.sleep(1.5 * (2 ** attempt))

    raise WeatherServiceError(f"Weather API request failed after retries: {last_exc}") from last_exc


async def fetch_forecasts_batch(locations: list[tuple[str, float, float]], days: int = 3) -> dict[str, dict]:
    """Fetch multiple monitored locations in ONE Open-Meteo request.

    Open-Meteo supports comma-separated latitude/longitude lists and returns
    a list of forecast objects. This avoids hammering the public API with one
    request per city during the live-monitor startup poll.
    """
    if not locations:
        return {}

    latitudes = ",".join(str(lat) for _, lat, _ in locations)
    longitudes = ",".join(str(lon) for _, _, lon in locations)
    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max",
        "forecast_days": days,
        "timezone": "Asia/Kolkata",
    }

    last_exc = None
    for attempt in range(3):
        try:
            resp = await _get_client().get(settings.weather_api_base, params=params)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict):
                payload = [payload]
            result = {}
            for (name, lat, lon), data in zip(locations, payload):
                result[name] = data
                _cache_set(f"fc:{round(lat, 2)}:{round(lon, 2)}:{days}", data, _FORECAST_TTL)
            return result
        except (httpx.HTTPError, ValueError) as exc:
            last_exc = exc
            if attempt < 2:
                await asyncio.sleep(1.5 * (2 ** attempt))

    raise WeatherServiceError(f"Batch weather API request failed after retries: {last_exc}") from last_exc


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
