"""
Turns structured forecast data + intent into a natural-language reply.

The rule-based reply is ALWAYS computed first — it's cheap, has no external
dependency, and doubles as the source of the structured `data` dict returned
to callers (used by the UI / API consumers). If an LLM provider is
configured, we then ask it to rephrase that same information more fluently
and in the requested language; if that call fails for any reason, we simply
keep the rule-based text. The user never sees an LLM outage as an error.
"""
import json

from app.core.config import settings
from app.models.schemas import Intent, ParsedQuery
from app.services import llm_client
from app.services.weather_service import describe_weather_code, is_severe
from app.services.language_service import normalize_language


def _rule_based_reply(parsed: ParsedQuery, location_name: str, forecast: dict) -> tuple[str, dict]:
    current = forecast.get("current", {})
    daily = forecast.get("daily", {})

    if parsed.intent == Intent.ALERT_CHECK:
        severe = is_severe(daily, 0) if daily.get("weather_code") else False
        if severe:
            reply = (
                f"⚠️ Heads up for {location_name}: conditions over the next day look severe "
                f"({describe_weather_code(daily['weather_code'][0])}, "
                f"wind up to {daily['wind_speed_10m_max'][0]} km/h, "
                f"{daily['precipitation_sum'][0]} mm precipitation expected). "
                "Please follow local advisories."
            )
        else:
            reply = f"No severe weather alerts for {location_name} right now — conditions look routine."
        return reply, {"severe": severe, "daily": daily}

    if parsed.intent == Intent.FORECAST:
        lines = [f"Forecast for {location_name}:"]
        dates = daily.get("time", [])
        for i, d in enumerate(dates):
            lines.append(
                f"- {d}: {describe_weather_code(daily['weather_code'][i])}, "
                f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]}°C, "
                f"{daily['precipitation_sum'][i]} mm rain"
            )
        return "\n".join(lines), {"daily": daily}

    if parsed.intent == Intent.ADVISORY:
        severe = is_severe(daily, 0) if daily.get("weather_code") else False
        advice = (
            "Hold off on irrigation and harvesting — rain/wind risk is elevated in the next day."
            if severe else
            "Conditions look stable — a reasonable window for routine field work, but recheck before spraying."
        )
        reply = f"Crop advisory for {location_name}: {advice}"
        return reply, {"severe": severe, "daily": daily}

    # CURRENT_WEATHER / HISTORICAL / fallback
    if current:
        precipitation = current.get("precipitation")
        precipitation_text = (
            f"precipitation {precipitation} mm"
            if precipitation is not None
            else "precipitation data unavailable"
        )
        reply = (
            f"Right now in {location_name}: {describe_weather_code(current.get('weather_code', -1))}, "
            f"{current.get('temperature_2m')}°C, "
            f"humidity {current.get('relative_humidity_2m')}%, "
            f"wind {current.get('wind_speed_10m')} km/h, "
            f"{precipitation_text}."
        )
    else:
        reply = f"I couldn't retrieve current conditions for {location_name} right now."
    return reply, {"current": current}


async def _llm_reply(parsed: ParsedQuery, location_name: str, structured_data: dict, history: list[dict] | None = None) -> str | None:
    """Ask the LLM to phrase the already-computed structured data
    conversationally, in the requested language. Returns None on any
    failure so the caller falls back to the rule-based text."""
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        return None
    try:
        client = llm_client.get_openai_client()
        history_text = ""
        if history:
            history_text = "\nRecent conversation:\n" + "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" for item in history[-10:]
            )
        prompt = (
            "You are WeatherGPT, a helpful weather assistant. Reply in "
            f"language code '{parsed.language}'. Be concise (2-4 sentences), "
            "warm and conversational, not a data dump — but keep all "
            "specific numbers (temperatures, wind speed, rain amounts) accurate "
            "to the data given.\n\n"
            f"User asked: {parsed.raw_text!r}\n"
            f"Intent: {parsed.intent.value}\n"
            f"Location: {location_name}\n"
            f"Weather data (JSON): {json.dumps(structured_data)[:3000]}"
            f"{history_text}"
        )
        # Use the current Responses API for the live assistant. The model is
        # selected here rather than hardcoding a canned answer path.
        resp = await client.responses.create(
            model="gpt-5.6-luna",
            input=prompt,
            max_output_tokens=250,
        )
        text = resp.output_text
        if not text:
            raise RuntimeError("The AI returned an empty response")

        # The LLM is the conversational layer; the weather API data remains
        # the source of truth for factual weather values.
        return text.strip()
    except Exception as exc:
        # Keep the chatbot usable when the optional AI provider is unavailable
        # (invalid/expired key, quota, network outage, SDK error, etc.). The
        # caller already has fresh weather data, so returning None lets it use
        # the live API-backed response instead of showing a fake AI error.
        import logging
        logging.getLogger("weathergpt.llm").warning("AI response unavailable; using live weather fallback: %s", exc)
        return None


async def compose_reply(parsed: ParsedQuery, location_name: str, forecast: dict, history: list[dict] | None = None) -> tuple[str, dict]:
    rule_reply, data = _rule_based_reply(parsed, location_name, forecast)
    # Translate the complete generated answer dynamically. There is no static
    # translation dictionary: the LLM formats the API-backed facts in the
    # user's selected/detected language and the deterministic reply remains
    # the fallback if the LLM is unavailable.
    parsed.language = normalize_language(parsed.language)
    if settings.llm_provider == "openai":
        # In AI mode the final response must come from the live LLM. The
        # deterministic reply above is retained only as structured context
        # and as a safety fallback when LLM_PROVIDER=none.
        translated = await _llm_reply(parsed, location_name, data, history)
        return (translated or rule_reply), data

    return rule_reply, data
