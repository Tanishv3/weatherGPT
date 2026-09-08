from fastapi import APIRouter, HTTPException, Query

from app.services.weather_service import fetch_forecast, describe_weather_code, is_severe, WeatherServiceError

router = APIRouter()


@router.get("/weather/here")
async def weather_here(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Current conditions for raw coordinates — built for the browser's
    Geolocation API, so it skips NLU/geocoding entirely (we already have
    exact coordinates, no place name to resolve or intent to parse)."""
    try:
        forecast = await fetch_forecast(lat, lon)
    except WeatherServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))

    current = forecast.get("current", {})
    daily = forecast.get("daily", {})

    return {
        "latitude": lat,
        "longitude": lon,
        "current": current,
        "condition": describe_weather_code(current.get("weather_code", -1)),
        "severe": is_severe(daily, 0) if daily.get("weather_code") else False,
        "today": {
            "temp_min": daily.get("temperature_2m_min", [None])[0],
            "temp_max": daily.get("temperature_2m_max", [None])[0],
            "precipitation_mm": daily.get("precipitation_sum", [None])[0],
        },
    }
