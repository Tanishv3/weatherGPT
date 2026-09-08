# WeatherGPT — backend starter

A working vertical slice of the conversational weather assistant: chat → intent
parsing → real forecast data → natural-language reply. Runs with **zero API
keys** (uses Open-Meteo, which is free/keyless, and a rule-based NLU
fallback) so it's demoable immediately, and upgrades smoothly once an LLM key
is added.

## Automated live monitoring (new)

The app now proactively watches a fixed set of major Indian regions —
**Goa, Mumbai, Pune, Delhi, Bengaluru, Chennai, Hyderabad, Kolkata,
Bhubaneswar, Ahmedabad, Jaipur, Thiruvananthapuram, Guwahati, Lucknow** —
instead of only responding when asked:

- `app/services/live_monitor.py` — background `asyncio` task, started at
  app startup, polls every `ALERT_POLL_INTERVAL_SECONDS` (default 300s).
  On each poll it updates an in-memory snapshot, and if a location
  transitions from normal → severe (reusing the existing `is_severe`
  heuristic — heavy rain/storm/high wind), it broadcasts an alert.
- `app/services/connection_manager.py` — tracks active WebSocket
  connections so the monitor can push to everyone connected, not just
  reply to whoever asked.
- `GET /live/weather` — REST snapshot of all monitored regions, for a
  dashboard that doesn't need a WebSocket at all.
- The chat UI (`/`) now shows a live-updating grid of all monitored cities
  (polls `/live/weather` every 60s) and displays pushed `alert` messages
  inline in the chat log — distinct styling, doesn't interrupt an
  in-progress chat reply.

To add/remove monitored locations, edit `MONITORED_LOCATIONS` in
`live_monitor.py` — it's just a list of `(name, lat, lon)` tuples.

Note: `fetch_forecast` caches per-location for ~10 min while the default
poll interval is 5 min, so the first couple of polls per location may
reuse the same cached data — Open-Meteo itself doesn't update much faster
than hourly anyway, so this doesn't meaningfully affect freshness. Lower
`weather_service._FORECAST_TTL` if you want tighter polling for a demo.

## Configuration

Copy `.env.example` to `.env` and fill in your keys — never commit `.env`:

```bash
cp .env.example .env
# then edit .env with your OPENAI_API_KEY
```

With `LLM_PROVIDER=openai` and a valid key set, both query understanding
(`app/services/nlu.py::_llm_parse`) and reply phrasing
(`app/services/composer.py::_llm_reply`) use the LLM, with the rule-based
logic kept as an automatic fallback if the call fails or no key is present —
nothing crashes either way. Gemini is read into config but not yet wired up.

## Run locally (no Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:
- **http://localhost:8000/** — interactive chat UI (WebSocket-based, streams a "thinking…" status before each reply, remembers the last location you asked about so follow-ups like "and tomorrow?" work).
- **http://localhost:8000/docs** — interactive API docs (Swagger).

## Run with Docker

```bash
docker compose up --build
```

## What changed for interactivity + performance

- **WebSocket chat (`/ws/chat`)** — real-time, bidirectional, with a lightweight per-connection session so you don't have to repeat the location every message, plus a "thinking" status event the UI uses to show a typing indicator. `POST /chat` still exists for stateless/one-shot integrations.
- **Chat UI (`/`)** — a minimal dependency-free HTML/JS page wired to the WebSocket endpoint, so there's something to actually click around in rather than only curl/Swagger. Root path no longer 404s.
- **Shared HTTP connection pool** — one `httpx.AsyncClient` is opened at app startup and reused for every outbound call to Open-Meteo (via FastAPI's `lifespan`), instead of opening a fresh TCP/TLS connection per request.
- **TTL caching** — geocoding results (effectively static) are cached ~24h; forecasts are cached ~10 min and keyed on a rounded lat/lon so nearby requests for the same city share a cache entry. Cuts repeat-query latency and outbound API load significantly.
- **GZip response compression** — cheap win on forecast JSON payloads and the frontend page.

## Try it

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "will it rain in Pune tomorrow?", "language": "en"}'
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "any cyclone alert for Chennai", "language": "en"}'
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "crop advisory for Nashik", "language": "en"}'
```

## What's implemented

- `app/services/nlu.py` — intent + location extraction. Uses OpenAI
  (JSON-mode structured output) when configured, with a rule-based fallback
  (regex + keyword matching) that runs when no key is set or the call fails.
- `app/services/composer.py` — turns structured forecast + intent into a
  reply. Rule-based text is always computed first (cheap, reliable, and used
  as the `data` field), then optionally rephrased more fluently by the LLM.
- `app/services/llm_client.py` — shared, lazily-initialized OpenAI client.
- `app/services/weather_service.py` — real forecast data via Open-Meteo,
  shared connection pool, TTL cache, plus a simple severity heuristic for alerts.
- `app/api/routes/chat.py` — stateless `POST /chat`.
- `app/api/routes/ws_chat.py` — real-time `WS /ws/chat` with per-connection
  session memory.
- `app/api/routes/alerts.py` — in-memory alert subscription (swap for a
  PostGIS table + MQTT fan-out for real geofenced push).
- `app/static/index.html` — minimal chat UI served at `/`.

## What's next (per the original architecture)

1. **Gemini support** — `settings.gemini_api_key` is read but unused; add a
   second branch in `llm_client.py` / the `_llm_parse` and `_llm_reply`
   provider checks if you want `LLM_PROVIDER=gemini` to actually work.
2. **Persistence** — SQLAlchemy models for users, query logs, and
   subscriptions (PostgreSQL + PostGIS for geofencing).
3. **Alert ingestion** — background worker polling WIS2.0/MQTT feeds,
   comparing against subscriptions, pushing via WebSocket/SMS.
4. **Voice** — STT/TTS at the client layer; backend already accepts plain
   text so it plugs in without backend changes.
5. **NWP integration** — swap/extend `weather_service.py` to pull directly
   from GFS/WRF model output for higher-resolution forecasts.

## A note on the API keys

The OpenAI/Gemini keys used here were shared in plaintext (chat + uploaded
file). Treat them as exposed regardless of how this project stores them —
**rotate both keys** in the OpenAI and Google AI consoles once you're done
testing. Going forward, keep secrets only in your local, gitignored `.env`
and avoid pasting them into chat tools or files that get shared.
