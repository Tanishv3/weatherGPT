from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "geocoding": "open-meteo",
        "country_filter": settings.default_country_code or "global",
    }
