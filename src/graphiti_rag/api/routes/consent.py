"""Privacy consent route.

``POST /api/consent`` records the user's data-sharing tier choice from the
privacy modal. The choice is held in :class:`AppState` (process-local; not
persisted across restarts) and influences how the LangSmith client is
constructed on the next pipeline build.

This endpoint is intentionally "shadow live" in clone-and-run builds: the
choice flows end-to-end through the wiring, but no traces are emitted
because no ``LANGSMITH_API_KEY`` is set in the end user's environment. The
endpoint and wiring become load-bearing the moment the app is hosted with
a real LangSmith key.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from graphiti_rag.api.state import app_state

logger = logging.getLogger(__name__)
router = APIRouter()


class ConsentPayload(BaseModel):
    tier: Literal["full", "anonymised", "metadata_only"]


@router.post("/consent")
async def set_consent(payload: ConsentPayload) -> dict[str, str]:
    await app_state.set_consent_tier(payload.tier)
    logger.info("Consent tier set to %s", payload.tier)
    return {"status": "ok", "tier": payload.tier}
