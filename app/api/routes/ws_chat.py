"""
Real-time chat over WebSocket. Unlike POST /chat (stateless, one-shot),
this keeps a short rolling history per connection so follow-ups like
"what about tomorrow?" can reuse the last mentioned location — and it lets
the client show a "typing"/status event before the final reply, which reads
as far more interactive than a plain request/response call.

Every connection also registers with the shared ConnectionManager, so the
background live_monitor task can push unsolicited "alert" events (e.g.
"rain developing in Goa") to it — this is what makes the app automated
rather than purely request/response.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.schemas import ChatRequest
from app.services.nlu import parse_query
from app.services.weather_service import geocode_location, fetch_forecast, WeatherServiceError
from app.services.composer import compose_reply
from app.services.connection_manager import manager

router = APIRouter()


# very small per-connection memory: last resolved location, so "and tomorrow?"
# works without repeating the place name every turn.
class Session:
    def __init__(self) -> None:
        self.last_location_text: str | None = None
        self.last_lat: float | None = None
        self.last_lon: float | None = None


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    manager.connect(websocket)
    session = Session()
    try:
        while True:
            payload = await websocket.receive_json()
            req = ChatRequest(**payload)

            # interactive "status" event so the client can show a typing indicator
            await manager.send(websocket, {"type": "status", "status": "thinking"})

            parsed = await parse_query(req.message, req.language)

            lat, lon = req.latitude, req.longitude
            location_name = "your location"
            try:
                if lat is None or lon is None:
                    location_text = parsed.location_text or session.last_location_text
                    if not location_text:
                        await manager.send(websocket, {
                            "type": "error",
                            "message": "Which location did you mean?",
                        })
                        continue
                    lat, lon, location_name = await geocode_location(location_text)
                else:
                    location_name = parsed.location_text or "your location"

                session.last_location_text = location_name
                session.last_lat, session.last_lon = lat, lon

                forecast_days = 7
                if parsed.intent.value == "forecast" and parsed.date_from and parsed.date_to:
                    forecast_days = max(1, min((parsed.date_to - parsed.date_from).days + 1, 16))
                forecast = await fetch_forecast(lat, lon, days=forecast_days)
            except WeatherServiceError as e:
                await manager.send(websocket, {"type": "error", "message": str(e)})
                continue

            reply, data = await compose_reply(parsed, location_name, forecast)

            await manager.send(websocket, {
                "type": "reply",
                "reply": reply,
                "intent": parsed.intent,
                "data": data,
                "language": req.language,
            })
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
