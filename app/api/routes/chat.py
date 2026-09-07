from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.nlu import parse_query
from app.services.weather_service import geocode_location, fetch_forecast, WeatherServiceError
from app.services.composer import compose_reply

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    parsed = await parse_query(req.message, req.language)

    # Resolve location: prefer explicit coordinates from the client, else
    # geocode free text extracted from the message.
    lat, lon = req.latitude, req.longitude
    location_name = "your location"
    try:
        if lat is None or lon is None:
            if not parsed.location_text:
                raise HTTPException(
                    status_code=422,
                    detail="Could not determine a location — please specify a place or share coordinates.",
                )
            lat, lon, location_name = await geocode_location(parsed.location_text)
        else:
            location_name = parsed.location_text or "your location"

        forecast_days = 7
        if parsed.intent.value == "forecast" and parsed.date_from and parsed.date_to:
            forecast_days = max(1, min((parsed.date_to - parsed.date_from).days + 1, 16))
        forecast = await fetch_forecast(lat, lon, days=forecast_days)
    except WeatherServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))

    reply, data = await compose_reply(parsed, location_name, forecast)

    return ChatResponse(reply=reply, intent=parsed.intent, data=data, language=req.language)
