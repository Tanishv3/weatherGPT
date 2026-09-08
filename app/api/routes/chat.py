from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.nlu import parse_query
from app.services.weather_service import geocode_location, fetch_forecast, WeatherServiceError
from app.services.composer import compose_reply
from app.services.language_service import detect_language, normalize_language

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Explicit selector wins. "auto" detects the language from the message.
    requested_language = req.language.strip().lower()
    language = await detect_language(req.message) if requested_language in {"", "auto", "detect"} else normalize_language(requested_language)
    parsed = await parse_query(req.message, language)

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

        forecast = await fetch_forecast(lat, lon)
    except WeatherServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))

    reply, data = await compose_reply(parsed, location_name, forecast)

    return ChatResponse(reply=reply, intent=parsed.intent, data=data, language=language)
