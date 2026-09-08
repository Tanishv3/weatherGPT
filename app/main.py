from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.api.routes import chat, health, alerts, ws_chat, live, location, here, dashboard, i18n
from app.services import weather_service, llm_client, live_monitor

logger = logging.getLogger("weathergpt.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loud, unmissable startup diagnostic: whether the app actually sees a
    # usable key for the SELECTED provider. Every AI-dependent feature
    # (translation, non-English location parsing, LLM-phrased replies)
    # silently degrades to English/rule-based behavior when this is False —
    # with no per-request error, so this line is the fastest way to tell
    # "LLM not working" from "LLM not configured".
    active = llm_client.active_provider()
    env_file_found = os.path.isfile(".env")
    if active is not None:
        logger.info("Startup check: %s provider is configured — AI features are ACTIVE.", active)
    else:
        # Distinguish "the file isn't found/loaded" from "the file loaded but
        # doesn't actually contain the key" — without ever logging the key
        # itself. This is the difference between a pydantic-settings config
        # problem and a plain typo/missing-line problem in .env.
        key_env_var = "OPENAI_API_KEY" if settings.llm_provider == "openai" else "GEMINI_API_KEY"
        key_set = bool(settings.openai_api_key) if settings.llm_provider == "openai" else bool(settings.gemini_api_key)
        env_mentions_key = False
        if env_file_found:
            try:
                with open(".env", "r", encoding="utf-8") as f:
                    env_mentions_key = key_env_var in f.read()
            except Exception:
                pass
        logger.warning(
            "Startup check: %r provider is NOT active (key set=%s). "
            "Translation, non-English location parsing, and AI-phrased "
            "replies will silently fall back to English/rule-based. "
            ".env file found in current directory: %s. "
            "'%s' text found inside that .env file: %s. "
            "If the file is found but the key text is NOT found, .env is "
            "missing that line or has a typo in the variable name. If the key "
            "text IS found but the setting is still not set, the file isn't "
            "being loaded/parsed correctly.",
            settings.llm_provider, key_set, env_file_found, key_env_var, env_mentions_key,
        )

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
app.include_router(here.router, tags=["here"])
app.include_router(dashboard.router, tags=["dashboard"])
app.include_router(i18n.router, tags=["i18n"])

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    # Serves the interactive chat UI so hitting the root isn't a 404 anymore.
    return FileResponse("app/static/index.html")
