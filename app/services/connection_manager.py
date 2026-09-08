"""
Tracks active WebSocket chat connections so the background live-weather
monitor (see live_monitor.py) can push alerts to everyone connected, not
just reply to whoever asked. A per-connection lock serializes writes, since
a normal chat reply and a broadcasted alert could otherwise both try to
send on the same socket at once.
"""
import asyncio

from fastapi import WebSocket

from app.services.language_service import translate_texts


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[WebSocket, asyncio.Lock] = {}
        self._languages: dict[WebSocket, str] = {}

    def connect(self, websocket: WebSocket, language: str = "en") -> None:
        self._connections[websocket] = asyncio.Lock()
        self._languages[websocket] = language

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.pop(websocket, None)
        self._languages.pop(websocket, None)

    async def send(self, websocket: WebSocket, message: dict) -> None:
        lock = self._connections.get(websocket)
        if lock is None:
            return
        async with lock:
            await websocket.send_json(message)

    def set_language(self, websocket: WebSocket, language: str) -> None:
        if websocket in self._connections:
            self._languages[websocket] = language

    async def broadcast(self, message: dict) -> None:
        # Snapshot the connection list first — sending can trigger
        # disconnects that would otherwise mutate the dict mid-iteration.
        for websocket in list(self._connections.keys()):
            try:
                outgoing = dict(message)
                if isinstance(outgoing.get("message"), str):
                    language = self._languages.get(websocket, "en")
                    outgoing["message"] = (await translate_texts([outgoing["message"]], language))[0]
                await self.send(websocket, outgoing)
            except Exception:
                self.disconnect(websocket)


manager = ConnectionManager()
