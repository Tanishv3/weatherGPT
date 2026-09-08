"""
Query understanding engine.

Design: try the LLM provider first (via a tool-call / structured-output
prompt) if one is configured; otherwise fall back to lightweight rule-based
parsing. This means the API is fully runnable and demoable with zero LLM
keys, and gets smarter the moment a key is added — same interface either way.
"""
import re
import json
import logging
from datetime import date, timedelta

from app.models.schemas import Intent, ParsedQuery
from app.core.config import settings
from app.services import llm_client

logger = logging.getLogger("weathergpt.nlu")

# Indian states/UTs and common cities. State names map to their capital so
# queries such as "weather in Goa" or "rain in Maharashtra" always resolve
# inside India instead of being interpreted as similarly named foreign places.
INDIA_LOCATION_ALIASES = {
    "andhra pradesh": "Amaravati", "arunachal pradesh": "Itanagar",
    "assam": "Guwahati", "bihar": "Patna", "chhattisgarh": "Raipur",
    "goa": "Panaji", "gujarat": "Gandhinagar", "haryana": "Chandigarh",
    "himachal pradesh": "Shimla", "jharkhand": "Ranchi", "karnataka": "Bengaluru",
    "kerala": "Thiruvananthapuram", "madhya pradesh": "Bhopal",
    "maharashtra": "Mumbai", "manipur": "Imphal", "meghalaya": "Shillong",
    "mizoram": "Aizawl", "nagaland": "Kohima", "odisha": "Bhubaneswar",
    "orissa": "Bhubaneswar", "punjab": "Chandigarh", "rajasthan": "Jaipur",
    "sikkim": "Gangtok", "tamil nadu": "Chennai", "telangana": "Hyderabad",
    "tripura": "Agartala", "uttar pradesh": "Lucknow", "uttarakhand": "Dehradun",
    "west bengal": "Kolkata",
    "andaman and nicobar islands": "Port Blair", "andaman": "Port Blair",
    "chandigarh": "Chandigarh", "dadra and nagar haveli and daman and diu": "Daman",
    "daman and diu": "Daman", "delhi": "New Delhi", "new delhi": "New Delhi",
    "jammu and kashmir": "Srinagar", "ladakh": "Leh",
    "lakshadweep": "Kavaratti", "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "mumbai": "Mumbai", "bombay": "Mumbai", "pune": "Pune", "nagpur": "Nagpur",
    "nashik": "Nashik", "nashik city": "Nashik", "thane": "Thane",
    "aurangabad": "Chhatrapati Sambhajinagar", "chhatrapati sambhajinagar": "Chhatrapati Sambhajinagar",
    "kolhapur": "Kolhapur", "goa city": "Panaji", "panaji": "Panaji",
    "vasco da gama": "Vasco da Gama", "ahmedabad": "Ahmedabad", "surat": "Surat",
    "vadodara": "Vadodara", "rajkot": "Rajkot", "gandhinagar": "Gandhinagar",
    "jaipur": "Jaipur", "jodhpur": "Jodhpur", "udaipur": "Udaipur",
    "kota": "Kota", "delhi ncr": "New Delhi", "noida": "Noida",
    "gurgaon": "Gurugram", "gurugram": "Gurugram", "lucknow": "Lucknow",
    "kanpur": "Kanpur", "agra": "Agra", "varanasi": "Varanasi", "prayagraj": "Prayagraj",
    "meerut": "Meerut", "chandigarh": "Chandigarh", "amritsar": "Amritsar",
    "ludhiana": "Ludhiana", "jalandhar": "Jalandhar", "shimla": "Shimla",
    "dehradun": "Dehradun", "haridwar": "Haridwar", "srinagar": "Srinagar",
    "jammu": "Jammu", "leh": "Leh", "bhopal": "Bhopal", "indore": "Indore",
    "gwalior": "Gwalior", "jabalpur": "Jabalpur", "raipur": "Raipur",
    "bilaspur": "Bilaspur", "ranchi": "Ranchi", "jamshedpur": "Jamshedpur",
    "patna": "Patna", "gaya": "Gaya", "kolkata": "Kolkata", "howrah": "Howrah",
    "siliguri": "Siliguri", "bhubaneswar": "Bhubaneswar", "cuttack": "Cuttack",
    "guwahati": "Guwahati", "dibrugarh": "Dibrugarh", "itanagar": "Itanagar",
    "gangtok": "Gangtok", "shillong": "Shillong", "imphal": "Imphal", "aizawl": "Aizawl",
    "kohima": "Kohima", "hyderabad": "Hyderabad", "warangal": "Warangal",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "mysuru": "Mysuru",
    "mangalore": "Mangaluru", "mangaluru": "Mangaluru", "chennai": "Chennai",
    "coimbatore": "Coimbatore", "madurai": "Madurai", "kochi": "Kochi",
    "ernakulam": "Kochi", "thiruvananthapuram": "Thiruvananthapuram", "trivandrum": "Thiruvananthapuram",
    "kozhikode": "Kozhikode", "vijayawada": "Vijayawada", "visakhapatnam": "Visakhapatnam",
    "vizag": "Visakhapatnam", "amaravati": "Amaravati", "tirupati": "Tirupati",
    "agra": "Agra", "port blair": "Port Blair", "puducherry": "Puducherry",
}

_FORECAST_WORDS = ("forecast", "tomorrow", "next", "week", "will it rain", "rain kal")
_ALERT_WORDS = ("alert", "warning", "cyclone", "flood", "storm", "danger")
_ADVISORY_WORDS = ("advisory", "crop", "sow", "irrigat", "harvest", "farmer")
_HISTORICAL_WORDS = ("last year", "history", "historical", "trend", "average", "compare")

# crude "in <location>" / "at <location>" extractor for the rule-based fallback.
# Deliberately greedy up to punctuation/end-of-string — the greedy capture is
# then trimmed of trailing non-location words (see _strip_trailing_stopwords),
# so "in Pune tomorrow?" resolves to "Pune", not "Pune tomorrow".
_LOCATION_PATTERN = re.compile(r"\b(?:in|at|for|near)\s+([A-Za-z][A-Za-z\s]{1,40})", re.IGNORECASE)

# Words that can legitimately trail a location mention but aren't part of the
# place name itself. Built from the intent keyword lists plus common time/
# filler words seen in casual weather queries.
_TRAILING_STOPWORDS = {
    w.split()[0] for w in (*_FORECAST_WORDS, *_ALERT_WORDS, *_ADVISORY_WORDS, *_HISTORICAL_WORDS)
} | {
    "today", "tonight", "now", "please", "right", "this", "weekend",
    "morning", "afternoon", "evening", "currently", "kal", "abhi",
}


def _strip_trailing_stopwords(location_text: str) -> str:
    """Repeatedly drop trailing words that are time/intent keywords rather
    than part of the place name, e.g. 'Pune tomorrow' -> 'Pune'."""
    words = location_text.strip().strip("?.,!").split()
    while words and words[-1].lower() in _TRAILING_STOPWORDS:
        words.pop()
    return " ".join(words)


def _normalize_indian_location(location_text: str | None) -> str | None:
    if not location_text:
        return None
    clean = re.sub(r"\s+", " ", location_text.strip().strip("?.,! "))
    key = clean.lower()
    if key in INDIA_LOCATION_ALIASES:
        return INDIA_LOCATION_ALIASES[key]
    # Also handle phrases like "Pune, Maharashtra" by checking the city first.
    first = key.split(",")[0].strip()
    return INDIA_LOCATION_ALIASES.get(first, clean)


def _rule_based_parse(text: str, language: str) -> ParsedQuery:
    lowered = text.lower()

    if any(w in lowered for w in _ALERT_WORDS):
        intent = Intent.ALERT_CHECK
    elif any(w in lowered for w in _ADVISORY_WORDS):
        intent = Intent.ADVISORY
    elif any(w in lowered for w in _HISTORICAL_WORDS):
        intent = Intent.HISTORICAL
    elif any(w in lowered for w in _FORECAST_WORDS):
        intent = Intent.FORECAST
    else:
        intent = Intent.CURRENT_WEATHER

    location_match = _LOCATION_PATTERN.search(text)
    location_text = None
    if location_match:
        stripped = _strip_trailing_stopwords(location_match.group(1))
        location_text = _normalize_indian_location(stripped or None)

    date_from = date.today()
    date_to = date_from + timedelta(days=3) if intent == Intent.FORECAST else date_from

    return ParsedQuery(
        intent=intent,
        location_text=location_text,
        date_from=date_from,
        date_to=date_to,
        language=language,
        raw_text=text,
    )


async def _llm_parse(text: str, language: str) -> ParsedQuery | None:
    """LLM-based query understanding via OpenAI JSON-mode structured output.

    Falls back to the rule-based parser (by returning None) if no provider/
    key is configured, or if the call fails for any reason — a flaky LLM
    call should never take the whole app down.
    """
    provider = llm_client.active_provider()
    if provider is None:
        return None

    system_prompt = (
        "You extract structured info from a weather-related user query, which "
        "may be written in any Indian language or script. "
        "Respond with ONLY a JSON object, no other text, with exactly these keys:\n"
        '- "intent": one of "current_weather", "forecast", "alert_check", '
        '"advisory", "historical", "unknown"\n'
        '- "location_text": the place name mentioned, TRANSLITERATED to its '
        "standard English/Latin-script spelling regardless of the query's "
        "language or script (e.g. a query containing \"पुणे\" or \"பூனா\" "
        'should still produce "Pune"), with NO time/date words in it (e.g. '
        '"Pune" not "Pune tomorrow"). null if no place is mentioned.\n'
        '- "days_ahead": integer 0-7, how many days forward the query refers '
        'to (0 = now/today).'
    )

    try:
        if provider == "openai":
            client = llm_client.get_openai_client()
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content or "{}"
        else:  # gemini
            raw = await llm_client.gemini_generate(
                f"{system_prompt}\n\nUser query: {text!r}",
                json_mode=True,
                max_output_tokens=150,
            )
        parsed_json = json.loads(raw)

        try:
            intent = Intent(parsed_json.get("intent", "unknown"))
        except ValueError:
            intent = Intent.UNKNOWN

        days_ahead = int(parsed_json.get("days_ahead") or 0)
        date_from = date.today()
        span = max(days_ahead, 3) if intent == Intent.FORECAST else days_ahead
        date_to = date_from + timedelta(days=span)

        # Never let the LLM invent/correct the user's location.
        # Extract the place directly from the original user message so
        # "rain in Goa" cannot become "Genoa" due to an LLM interpretation.
        rule_parsed = _rule_based_parse(text, language)
        location_text = rule_parsed.location_text or _normalize_indian_location(parsed_json.get("location_text")) or None

        return ParsedQuery(
            intent=intent,
            location_text=location_text,
            date_from=date_from,
            date_to=date_to,
            language=language,
            raw_text=text,
        )
    except Exception as exc:
        # Network error, rate limit, malformed JSON, etc. — degrade gracefully,
        # but log it: a silent fallback here is what makes non-English
        # location parsing quietly break, and this is the only place that
        # explains why.
        logger.warning("nlu._llm_parse failed, falling back to rule-based parsing: %s", exc)
        return None


async def parse_query(text: str, language: str = "en") -> ParsedQuery:
    parsed = await _llm_parse(text, language)
    if parsed is not None:
        return parsed
    return _rule_based_parse(text, language)
