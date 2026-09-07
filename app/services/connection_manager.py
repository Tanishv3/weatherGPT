"""
Tracks active WebSocket chat connections so the background live-weather
monitor (see live_monitor.py) can push alerts to everyone connected, not
just reply to whoever asked. A per-connection lock serializes writes, since
a normal chat reply and a broadcasted alert could otherwise both try to
send on the same socket at once.
"""
import asyncio

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[WebSocket, asyncio.Lock] = {}

    def connect(self, websocket: WebSocket) -> None:
        self._connections[websocket] = asyncio.Lock()

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.pop(websocket, None)

    async def send(self, websocket: WebSocket, message: dict) -> None:
        lock = self._connections.get(websocket)
        if lock is None:
            return
        async with lock:
            await websocket.send_json(message)

    async def broadcast(self, message: dict) -> None:
        # Snapshot the connection list first — sending can trigger
        # disconnects that would otherwise mutate the dict mid-iteration.
        for websocket in list(self._connections.keys()):
            try:
                await self.send(websocket, message)
            except Exception:
                self.disconnect(websocket)


manager = ConnectionManager()
