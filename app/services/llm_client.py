"""
Shared LLM client(s), following the same reuse pattern as
weather_service.py's HTTP client — one client instance, not one per request.

Two providers are wired up:
- OpenAI, via the official SDK (AsyncOpenAI).
- Gemini, via a plain httpx call to the generateContent REST endpoint —
  no extra SDK dependency needed, and it mirrors the pooled-client pattern
  already used for the weather API.

`active_provider()` is the single source of truth for "is there actually a
usable AI provider right now" (selected provider AND a matching key
present). nlu.py, language_service.py and composer.py all call it instead
of re-deriving the same condition, so there's exactly one place that
decides when AI features are active vs. silently falling back to
rule-based/English behavior.
"""
import asyncio
import httpx
import os
from openai import OpenAI
try:
    from openai import AsyncOpenAI
except ImportError:  # OpenAI is optional when LLM_PROVIDER=none
    AsyncOpenAI = None  # type: ignore[assignment,misc]

from app.core.config import settings

_openai_client = None
_gemini_client: httpx.AsyncClient | None = None

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# gemini-2.5-flash was deprecated and is being shut down through 2026 — Google
# recommends gemini-3.x Flash models now. gemini-3.6-flash is the current
# stable/GA pick as of this writing (newer 3.7/3.8 previews exist too, but
# 3.6 has been GA the longest and is the safer default).
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"


def active_provider() -> str | None:
    """Which AI provider is actually usable given current config, or None
    if AI features should fall back to rule-based/English behavior."""
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return "openai"
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        return "gemini"
    return None


def get_openai_client():
    global _openai_client
    if AsyncOpenAI is None:
        raise RuntimeError("OpenAI package is not installed")
    if _openai_client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def get_gemini_client() -> httpx.AsyncClient:
    global _gemini_client
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    if _gemini_client is None:
        _gemini_client = httpx.AsyncClient(timeout=httpx.Timeout(20.0))
    return _gemini_client


async def gemini_generate(
    prompt: str,
    *,
    model: str = GEMINI_DEFAULT_MODEL,
    json_mode: bool = False,
    max_output_tokens: int = 512,
    temperature: float = 0.0,
    thinking_level: str = "low",
) -> str:
    """Call Gemini's generateContent REST endpoint and return the text of
    the first candidate. Raises on any failure (bad key, quota, network,
    empty response) — callers catch broadly and fall back to rule-based
    logic, same pattern used around the OpenAI calls.

    thinking_level defaults to "low": Gemini 3.x models think by default
    (thinkingLevel="medium"), and those reasoning tokens are deducted from
    the SAME max_output_tokens budget as the actual answer. For short
    structured-output calls (JSON translation payloads, NLU extraction)
    that easily ate the whole budget on reasoning and left the real answer
    truncated mid-JSON — which is what caused "Unterminated string" /
    "Expecting value" json.loads failures upstream. Pass thinking_level=
    None to use the model's default instead."""
    client = get_gemini_client()
    generation_config: dict = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    if thinking_level:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    url = f"{GEMINI_API_BASE}/{model}:generateContent"

    # 503 UNAVAILABLE ("high demand") and other transient 5xx errors are
    # usually a few-second blip on Google's side, not a real failure — a
    # couple of quick retries clears most of them without the caller ever
    # noticing. 429 (quota) is deliberately NOT retried here: the quota
    # reset can be tens of seconds away, which would just make the user
    # wait instead of getting the (perfectly fine) rule-based fallback.
    for attempt in range(3):
        resp = await client.post(url, params={"key": settings.gemini_api_key}, json=body)
        if resp.status_code < 400:
            break
        if resp.status_code in (500, 502, 503, 504) and attempt < 2:
            await asyncio.sleep(0.6 * (attempt + 1))
            continue
        # Surface Google's actual error body (e.g. "model not found for API
        # version v1beta") instead of just the bare status code — that
        # detail is what actually explains *why* a call failed.
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        # e.g. blocked by safety filters — surface something informative
        # rather than a bare KeyError.
        feedback = data.get("promptFeedback")
        raise RuntimeError(f"Gemini returned no candidates (promptFeedback={feedback})")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text


async def close_clients() -> None:
    global _openai_client, _gemini_client
    if _openai_client is not None:
        await _openai_client.close()
        _openai_client = None
    if _gemini_client is not None:
        await _gemini_client.aclose()
        _gemini_client = None
