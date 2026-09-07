from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    CURRENT_WEATHER = "current_weather"
    FORECAST = "forecast"
    ALERT_CHECK = "alert_check"
    ADVISORY = "advisory"          # e.g. crop advisory
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class ParsedQuery(BaseModel):
    intent: Intent
    location_text: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    date_from: date | None = None
    date_to: date | None = None
    language: str = "en"
    raw_text: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    language: str = "en"          # ISO code: en, hi, mr, ta, te, bn, ...
    latitude: float | None = None
    longitude: float | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    intent: Intent
    data: dict | None = None      # raw structured data behind the reply, for UI/debug
    language: str = "en"


class AlertSubscription(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    radius_km: float = 25.0
    channels: list[str] = ["push"]  # push | sms | voice
