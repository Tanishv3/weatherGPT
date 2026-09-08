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
        reply = (
            f"Right now in {location_name}: {describe_weather_code(current.get('weather_code', -1))}, "
            f"{current.get('temperature_2m')}°C, "
            f"humidity {current.get('relative_humidity_2m')}%, "
            f"wind {current.get('wind_speed_10m')} km/h."
        )
    else:
        reply = f"I couldn't retrieve current conditions for {location_name} right now."
    return reply, {"current": current}


async def _llm_reply(parsed: ParsedQuery, location_name: str, structured_data: dict) -> str | None:
    """Ask the LLM to phrase the already-computed structured data
    conversationally, in the requested language. Returns None on any
    failure so the caller falls back to the rule-based text."""
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        return None
    try:
        client = llm_client.get_openai_client()
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
        )
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=250,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None
    except Exception:
        # Network error, rate limit, auth failure, etc. — degrade gracefully.
        return None


async def compose_reply(parsed: ParsedQuery, location_name: str, forecast: dict) -> tuple[str, dict]:
    rule_reply, data = _rule_based_reply(parsed, location_name, forecast)
    llm_reply = await _llm_reply(parsed, location_name, data)
    return (llm_reply or rule_reply), data
