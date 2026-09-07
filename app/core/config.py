"""Central application configuration loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "WeatherGPT"
    app_version: str = "2.1-india-first-dynamic"

    database_url: str = "postgresql+asyncpg://weathergpt:weathergpt@db:5432/weathergpt"

    llm_provider: str = "openai"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash"

    weather_api_base: str = "https://api.open-meteo.com/v1/forecast"
    geocoding_api_base: str = "https://geocoding-api.open-meteo.com/v1/search"

    # India-first geocoding. Set to empty/null only if you intentionally want
    # global location matching.
    default_country_code: str = "IN"

    alert_poll_interval_seconds: int = 300

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
