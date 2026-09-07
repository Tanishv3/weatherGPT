from fastapi import APIRouter

from app.models.schemas import AlertSubscription

router = APIRouter()

# In-memory demo store. Replace with a PostGIS-backed table
# (see app/db/models.py) for real geofenced push in production.
_subscriptions: dict[str, AlertSubscription] = {}


@router.post("/alerts/subscribe")
async def subscribe(sub: AlertSubscription):
    _subscriptions[sub.user_id] = sub
    return {"status": "subscribed", "active_subscriptions": len(_subscriptions)}


@router.get("/alerts/subscriptions")
async def list_subscriptions():
    return list(_subscriptions.values())
