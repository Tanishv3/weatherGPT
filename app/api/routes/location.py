from fastapi import APIRouter, HTTPException, Query

from app.services.weather_service import WeatherServiceError, geocode_location

router = APIRouter()


@router.get("/location/resolve")
async def resolve_location(name: str = Query(..., min_length=1, max_length=100)):
    """Resolve a place using the same dynamic India-first resolver as chat."""
    try:
        lat, lon, resolved_name = await geocode_location(name)
    except WeatherServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "query": name,
        "resolved_name": resolved_name,
        "latitude": lat,
        "longitude": lon,
    }
