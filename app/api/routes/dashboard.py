from fastapi import APIRouter, HTTPException, Query

from app.services.weather_service import (
    WeatherServiceError,
    describe_weather_code,
    fetch_air_quality,
    fetch_forecast,
    geocode_location,
    reverse_geocode_location,
    is_severe,
    weather_icon,
)
from app.services.live_monitor import get_snapshot, MONITORED_LOCATIONS

router = APIRouter()


@router.get("/dashboard/weather")
async def dashboard_weather(
    city: str | None = Query(None, description="Place name to search for, e.g. 'Chennai'"),
    lat: float | None = Query(None, ge=-90, le=90),
    lon: float | None = Query(None, ge=-180, le=180),
):
    """Everything the dashboard's main view needs in one call: resolved
    location name, current conditions, a 7-day daily forecast, and
    best-effort AQI. Pass either `city` (geocoded, India-first) or `lat`/`lon`
    (e.g. from the browser's Geolocation API) — not both."""
    if city:
        try:
            latitude, longitude, resolved_name = await geocode_location(city)
        except WeatherServiceError as e:
            raise HTTPException(status_code=502, detail=str(e))
    elif lat is not None and lon is not None:
        latitude, longitude = lat, lon
        resolved_name = await reverse_geocode_location(latitude, longitude)
    else:
        raise HTTPException(status_code=422, detail="Provide either `city` or both `lat` and `lon`.")

    try:
        forecast = await fetch_forecast(latitude, longitude, days=7)
    except WeatherServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))

    aqi = await fetch_air_quality(latitude, longitude)  # best-effort, may be None

    current = forecast.get("current", {})
    daily = forecast.get("daily", {})
    daily_len = len(daily.get("time", []))

    days = [
        {
            "date": daily["time"][i],
            "temp_min": daily["temperature_2m_min"][i],
            "temp_max": daily["temperature_2m_max"][i],
            "precipitation_mm": daily["precipitation_sum"][i],
            "rain_chance_pct": daily.get("precipitation_probability_max", [None] * daily_len)[i],
            "condition": describe_weather_code(daily["weather_code"][i]),
            "icon": weather_icon(daily["weather_code"][i]),
        }
        for i in range(daily_len)
    ]

    return {
        "resolved_name": resolved_name,
        "latitude": latitude,
        "longitude": longitude,
        "current": {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_kmh": current.get("wind_speed_10m"),
            "pressure_hpa": current.get("surface_pressure"),
            "precipitation_mm": current.get("precipitation"),
            "condition": describe_weather_code(current.get("weather_code", -1)),
            "icon": weather_icon(current.get("weather_code", -1)),
        },
        "severe": is_severe(daily, 0) if daily.get("weather_code") else False,
        "days": days,
        "aqi": aqi,  # None if unavailable — frontend should handle gracefully
    }


@router.get("/dashboard/other-cities")
async def dashboard_other_cities(exclude: str | None = Query(None)):
    """A handful of the automatically-monitored Indian cities (see
    live_monitor.py) for the 'Other cities' panel — reuses the same
    in-memory snapshot the live dashboard and alert system already
    maintain, so this costs nothing extra to serve."""
    snapshot = get_snapshot()
    cities = []
    for name, _, _ in MONITORED_LOCATIONS:
        if name == exclude or name not in snapshot:
            continue
        entry = snapshot[name]
        cur = entry.get("current", {})
        cities.append({
            "name": name,
            "temperature": cur.get("temperature_2m"),
            "condition": entry.get("today", {}).get("condition"),
            "severe": entry.get("severe", False),
        })
        if len(cities) >= 4:
            break
    return {"cities": cities}
