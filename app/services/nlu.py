"""Weather query understanding with deterministic location/date extraction.

The LLM is used only to improve intent understanding. Location extraction is
kept deterministic so a place such as "Goa" cannot accidentally be rewritten
as an unrelated place such as Genoa by the language model.
"""
import json
import re
from datetime import date, timedelta

from app.core.config import settings
from app.models.schemas import Intent, ParsedQuery
from app.services import llm_client

_FORECAST_WORDS = (
    "forecast", "tomorrow", "next", "week", "will it rain", "rain kal",
    "days", "day forecast", "weather forecast",
)
_ALERT_WORDS = ("alert", "warning", "cyclone", "flood", "storm", "danger")
_ADVISORY_WORDS = ("advisory", "crop", "sow", "irrigat", "harvest", "farmer")
_HISTORICAL_WORDS = ("last year", "history", "historical", "trend", "average", "compare")

# Location is extracted from the user's own message before the LLM runs.
# This prevents LLM hallucinations/typos from changing the requested place.
_LOCATION_PATTERNS = (
    # "forecast for Goa", "weather in Pune", "temperature at Nashik"
    re.compile(
        r"\b(?:in|at|for|near|of)\s+(.+?)(?="
        r"\s+(?:today|tonight|tomorrow|now|this|next|on|from|during|please|for\s+\d+\s+days?)\b|[?!.,;:]?$)",
        re.IGNORECASE,
    ),
    # "Goa weather" / "Goa forecast"
    re.compile(
        r"^\s*(.+?)\s+(?:weather|forecast|temperature|conditions)\s*$",
        re.IGNORECASE,
    ),
)

_GENERIC_LOCATION_STOPWORDS = {
    "weather", "forecast", "temperature", "conditions", "rain", "raining",
    "today", "tonight", "tomorrow", "now", "please", "currently", "current",
    "this", "next", "week", "weekend", "morning", "afternoon", "evening",
    "will", "be", "is", "the", "for", "me", "give", "show", "tell",
}

_DURATION_TAIL = re.compile(
    r"\s+(?:for|next)\s+(?:\d{1,2}\s*[- ]?days?|one|a|the\s+next)\s*(?:week)?\s*$",
    re.IGNORECASE,
)


def _extract_location(text: str) -> str | None:
    """Extract the user's literal place name before any LLM sees it.

    This is intentionally deterministic. The extracted text is what gets sent
    to the geocoder, so an LLM cannot turn "Goa" into "Genoa" or otherwise
    rewrite the requested location.
    """
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return None

    value: str | None = None
    for pattern in _LOCATION_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            value = match.group(1).strip(" \t\r\n?.,!;:")
            break

    if not value:
        return None

    # Remove forecast-duration tails such as "Goa for 7 days".
    value = _DURATION_TAIL.sub("", value)
    value = re.sub(r"\s+\d{1,2}\s*[- ]?days?\s*$", "", value, flags=re.I)
    value = re.sub(r"^\d{1,2}\s*[- ]?days?\s+(?:weather\s+)?forecast\s+(?:for\s+)?", "", value, flags=re.I)
    value = re.sub(r"\s+(?:weather|forecast|temperature|conditions)\s*$", "", value, flags=re.I)

    words = value.split()
    while words and words[-1].lower() in _GENERIC_LOCATION_STOPWORDS:
        words.pop()
    value = " ".join(words).strip(" ,")
    return value or None



def _requested_forecast_days(text: str) -> int | None:
    lowered = text.lower()
    match = re.search(r"\b(\d{1,2})\s*[- ]?day(?:s)?\b", lowered)
    if match:
        return max(1, min(int(match.group(1)), 16))
    if re.search(r"\b(?:one|a)\s+week\b|\b7\s*day\b|\bweek(?:ly)?\b", lowered):
        return 7
    if "tomorrow" in lowered:
        return 2
    return None


def _infer_intent(text: str) -> Intent:
    lowered = text.lower()
    if any(w in lowered for w in _ALERT_WORDS):
        return Intent.ALERT_CHECK
    if any(w in lowered for w in _ADVISORY_WORDS):
        return Intent.ADVISORY
    if any(w in lowered for w in _HISTORICAL_WORDS):
        return Intent.HISTORICAL
    if any(w in lowered for w in _FORECAST_WORDS) or _requested_forecast_days(text):
        return Intent.FORECAST
    return Intent.CURRENT_WEATHER


def _rule_based_parse(text: str, language: str) -> ParsedQuery:
    intent = _infer_intent(text)
    location_text = _extract_location(text)
    date_from = date.today()
    days = _requested_forecast_days(text)

    if intent == Intent.FORECAST:
        forecast_days = days or 3
        date_to = date_from + timedelta(days=forecast_days - 1)
    else:
        date_to = date_from

    return ParsedQuery(
        intent=intent,
        location_text=location_text,
        date_from=date_from,
        date_to=date_to,
        language=language,
        raw_text=text,
    )


async def _llm_parse(text: str, language: str) -> ParsedQuery | None:
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        return None

    deterministic_location = _extract_location(text)
    requested_days = _requested_forecast_days(text)

    try:
        client = llm_client.get_openai_client()
        system_prompt = (
            "Extract structured weather query information. Return ONLY JSON with exactly these keys: "
            '"intent" (current_weather|forecast|alert_check|advisory|historical|unknown), '
            '"location_text" (string or null), "forecast_days" (integer 1-16 or null). '
            "Preserve the user's location exactly; do not translate, autocorrect, or replace a place name. "
            "For example, Goa must remain Goa, not Genoa. "
            "For a request like '7 day forecast for Goa', forecast_days must be 7."
        )
        resp = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=150,
        )
        parsed_json = json.loads(resp.choices[0].message.content or "{}")
        try:
            intent = Intent(parsed_json.get("intent", "unknown"))
        except ValueError:
            intent = _infer_intent(text)

        # Deterministic extraction wins whenever it found an explicit location.
        location_text = deterministic_location or parsed_json.get("location_text") or None
        llm_days = parsed_json.get("forecast_days")
        try:
            llm_days = int(llm_days) if llm_days is not None else None
        except (TypeError, ValueError):
            llm_days = None
        forecast_days = requested_days or (max(1, min(llm_days, 16)) if llm_days else 3)

        date_from = date.today()
        date_to = date_from + timedelta(days=forecast_days - 1) if intent == Intent.FORECAST else date_from
        return ParsedQuery(
            intent=intent,
            location_text=location_text,
            date_from=date_from,
            date_to=date_to,
            language=language,
            raw_text=text,
        )
    except Exception:
        return None


async def parse_query(text: str, language: str = "en") -> ParsedQuery:
    parsed = await _llm_parse(text, language)
    if parsed is not None:
        return parsed
    return _rule_based_parse(text, language)
