"""Automated language detection and translation for WeatherGPT.

The application does not keep a hard-coded translation table. The configured
LLM is used to detect the user's language and translate UI/data text while the
selected language is cached in-process for efficiency.
"""
import json
import logging
import re
import time
from typing import Iterable

from app.core.config import settings
from app.services import llm_client

logger = logging.getLogger("weathergpt.language_service")

# UI options are restricted to languages used in India. English remains the
# internal fallback only; it is intentionally not exposed in the selector.
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
    "mr": "मराठी",
    "bn": "বাংলা",
    "te": "తెలుగు",
    "ta": "தமிழ்",
    "gu": "ગુજરાતી",
    "kn": "ಕನ್ನಡ",
    "ml": "മലയാളം",
    "pa": "ਪੰਜਾਬੀ",
    "or": "ଓଡ଼ିଆ",
    "as": "অসমীয়া",
    "ur": "اردو",
    "ne": "नेपाली",
    "kok": "कोंकणी",
    "mai": "मैथिली",
    "sa": "संस्कृतम्",
    "mni": "মৈতৈলোন্",
    "doi": "डोगरी",
    "ks": "कॉशुर / कश्मीरी",
    "sd": "सिन्धी",
    "sat": "संताली",
    "brx": "बड़ो",
}

_LANGUAGE_HINTS = {
    "hi": ("देवनागरी", "है", "का", "की", "मौसम"),
    "mr": ("आहे", "चा", "ची", "मध्ये", "महाराष्ट्र"),
    "bn": ("আছে", "এর", "আজ", "আবহাওয়া"),
    "gu": ("છે", "માં", "નું", "હવામાન"),
    "ta": ("உள்ளது", "இன்று", "மழை", "வானிலை"),
    "te": ("ఉంది", "లో", "ఈరోజు", "వాతావరణం"),
    "kn": ("ಇದೆ", "ನಲ್ಲಿ", "ಇಂದು", "ಹವಾಮಾನ"),
    "ml": ("ഉണ്ട്", "യിൽ", "ഇന്ന്", "കാലാവസ്ഥ"),
    "pa": ("ਹੈ", "ਵਿੱਚ", "ਅੱਜ", "ਮੌਸਮ"),
    "or": ("ଅଛି", "ରେ", "ଆଜି", "ପାଣିପାଗ"),
    "as": ("আছে", "আজি", "বতৰ", "মাজত"),
    "ur": ("ہے", "میں", "آج", "موسم"),
    "ne": ("छ", "मा", "आज", "मौसम"),
}

_cache: dict[tuple[str, str], tuple[float, str]] = {}
_CACHE_TTL = 3600


def normalize_language(code: str | None) -> str:
    code = (code or "en").strip().lower().replace("_", "-")
    code = code.split("-")[0]
    if code in SUPPORTED_LANGUAGES or code == "en":
        return code
    return "en"


def detect_language_local(text: str) -> str | None:
    """Fast fallback detection based on script when the LLM is unavailable."""
    if not text:
        return None
    counts = {
        "devanagari": len(re.findall(r"[\u0900-\u097F]", text)),
        "bengali": len(re.findall(r"[\u0980-\u09FF]", text)),
        "gurmukhi": len(re.findall(r"[\u0A00-\u0A7F]", text)),
        "gujarati": len(re.findall(r"[\u0A80-\u0AFF]", text)),
        "oriya": len(re.findall(r"[\u0B00-\u0B7F]", text)),
        "tamil": len(re.findall(r"[\u0B80-\u0BFF]", text)),
        "telugu": len(re.findall(r"[\u0C00-\u0C7F]", text)),
        "kannada": len(re.findall(r"[\u0C80-\u0CFF]", text)),
        "malayalam": len(re.findall(r"[\u0D00-\u0D7F]", text)),
        "arabic": len(re.findall(r"[\u0600-\u06FF]", text)),
        "meitei": len(re.findall(r"[\uABC0-\uABFF]", text)),
        "santali": len(re.findall(r"[\u1C50-\u1C7F]", text)),
    }
    script, score = max(counts.items(), key=lambda x: x[1])
    if score == 0:
        return None
    return {
        "devanagari": "hi", "bengali": "bn", "gurmukhi": "pa",
        "gujarati": "gu", "oriya": "or", "tamil": "ta", "telugu": "te",
        "kannada": "kn", "malayalam": "ml", "arabic": "ur", "meitei": "mni",
        "santali": "sat",
    }[script]


async def detect_language(text: str) -> str:
    local = detect_language_local(text)
    if local:
        return local
    provider = llm_client.active_provider()
    if provider is None:
        return "en"
    system_prompt = (
        "Identify the language of the user's message. Return ONLY its ISO 639-1 "
        "or ISO-compatible code. If it is an Indian language, use one of: "
        + ", ".join(SUPPORTED_LANGUAGES.keys())
        + ". Otherwise return en. Do not translate the message."
    )
    try:
        if provider == "openai":
            client = llm_client.get_openai_client()
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:1000]},
                ],
                temperature=0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content
        else:  # gemini
            raw = await llm_client.gemini_generate(
                f"{system_prompt}\n\nMessage: {text[:1000]!r}",
                max_output_tokens=10,
            )
        return normalize_language(raw)
    except Exception as exc:
        logger.warning("language_service.detect_language failed, defaulting to 'en': %s", exc)
        return "en"


async def translate_texts(texts: Iterable[str], language: str) -> list[str]:
    texts = list(texts)
    language = normalize_language(language)
    if not texts or language == "en":
        return texts

    result: list[str] = []
    missing: list[tuple[int, str]] = []
    now = time.time()
    for i, text in enumerate(texts):
        key = (language, text)
        cached = _cache.get(key)
        if cached and cached[0] > now:
            result.append(cached[1])
        else:
            result.append("")
            missing.append((i, text))

    provider = llm_client.active_provider()
    if not missing or provider is None:
        for i, text in missing:
            result[i] = text
        return result

    try:
        payload = [{"id": i, "text": text} for i, text in missing]
        prompt = (
            f"Translate each text into {SUPPORTED_LANGUAGES.get(language, language)}. "
            "Return ONLY a JSON object with a \"translations\" array; each item must contain id and translation. "
            "Preserve numbers, units, emojis, city names, punctuation and line breaks. "
            "Do not add explanations. Do not translate proper place names unless they "
            "normally have a native-language form.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        max_tokens = max(300, min(4000, len(missing) * 80))
        if provider == "openai":
            client = llm_client.get_openai_client()
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
            )
            raw_content = resp.choices[0].message.content or "{}"
        else:  # gemini
            raw_content = await llm_client.gemini_generate(
                prompt,
                json_mode=True,
                max_output_tokens=max_tokens,
            )
        raw = json.loads(raw_content)
        rows = raw.get("translations", raw if isinstance(raw, list) else [])
        by_id = {int(row["id"]): str(row["translation"]) for row in rows if "id" in row and "translation" in row}
        for i, text in missing:
            translated = by_id.get(i, text)
            result[i] = translated
            _cache[(language, text)] = (time.time() + _CACHE_TTL, translated)
        return result
    except Exception as exc:
        logger.warning("language_service.translate_texts failed for language=%s, returning source text: %s", language, exc)
        for i, text in missing:
            result[i] = text
        return result
