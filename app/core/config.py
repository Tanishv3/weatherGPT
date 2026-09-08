"""
Central configuration. All values are overridable via environment variables
so the same code runs locally, in Docker, and in k8s.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "WeatherGPT"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://weathergpt:weathergpt@db:5432/weathergpt"

    # --- LLM provider ---
    # If no key is set, the NLU/composer layer falls back to rule-based logic
    # so the app still runs end-to-end for a demo without any paid API.
    # Values are read from environment / .env — never hardcode secrets here.
    llm_provider: str = "openai"  # "openai" | "gemini" | "none"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    # --- Weather data source ---
    # Open-Meteo requires no API key and is good enough for a working demo;
    # swap for IMD/GFS-WRF outputs in production.
    weather_api_base: str = "https://api.open-meteo.com/v1/forecast"
    geocoding_api_base: str = "https://geocoding-api.open-meteo.com/v1/search"
    air_quality_api_base: str = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # --- Alerts ---
    alert_poll_interval_seconds: int = 600

    class Config:
        env_file = ".env"


settings = Settings()
