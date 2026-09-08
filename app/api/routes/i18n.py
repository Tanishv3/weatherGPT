from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.language_service import SUPPORTED_LANGUAGES, detect_language, normalize_language, translate_texts

router = APIRouter(prefix="/i18n")


class TranslateRequest(BaseModel):
    texts: list[str] = Field(default_factory=list, max_length=100)
    language: str = "en"


class TranslateResponse(BaseModel):
    translations: list[str]
    language: str


@router.get("/languages")
async def languages():
    return {"languages": [{"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES.items()]}


@router.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    language = normalize_language(req.language)
    return {"translations": await translate_texts(req.texts, language), "language": language}


@router.post("/detect")
async def detect(payload: dict):
    text = str(payload.get("text", ""))[:1000]
    return {"language": await detect_language(text)}
