from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.api.routes import chat, health, alerts, ws_chat, live, location
from app.services import weather_service, llm_client, live_monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open one pooled HTTP client for the whole app's lifetime
    # instead of a new connection per request (see weather_service.py).
    await weather_service.init_client()
    # Start the background task that automatically polls major Indian
    # regions and pushes alerts on newly-severe weather (see live_monitor.py).
    live_monitor.start()
    yield
    # Shutdown: release connection pools and stop the background task cleanly.
    await live_monitor.stop()
    await weather_service.close_client()
    await llm_client.close_clients()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before production
    allow_methods=["*"],
    allow_headers=["*"],
)
# Compress JSON/HTML responses over a size threshold — cheap win for
# forecast payloads and the frontend bundle.
app.add_middleware(GZipMiddleware, minimum_size=500)

app.include_router(health.router, tags=["health"])
app.include_router(chat.router, tags=["chat"])
app.include_router(alerts.router, tags=["alerts"])
app.include_router(ws_chat.router, tags=["chat-ws"])
app.include_router(live.router, tags=["live"])
app.include_router(location.router, tags=["location"])

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    # Serves the interactive chat UI so hitting the root isn't a 404 anymore.
    return FileResponse("app/static/index.html")
