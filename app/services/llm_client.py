"""
Shared LLM client(s), following the same reuse pattern as
weather_service.py's HTTP client — one client instance, not one per request.

Only OpenAI is fully wired today (it's the key format that was provided and
maps cleanly onto the JSON-mode structured-output flow used in nlu.py /
composer.py). The Gemini key is read into config for when that provider is
implemented; until then, llm_provider="gemini" simply falls through to the
rule-based logic rather than erroring — see the provider checks in
nlu._llm_parse and composer._llm_reply.
"""
try:
    from openai import AsyncOpenAI
except ImportError:  # OpenAI is optional when LLM_PROVIDER=none
    AsyncOpenAI = None  # type: ignore[assignment,misc]

from app.core.config import settings

_openai_client = None


def get_openai_client():
    global _openai_client
    if AsyncOpenAI is None:
        raise RuntimeError("OpenAI package is not installed")
    if _openai_client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


async def close_clients() -> None:
    global _openai_client
    if _openai_client is not None:
        await _openai_client.close()
        _openai_client = None
