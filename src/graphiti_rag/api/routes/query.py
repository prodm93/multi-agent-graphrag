"""POST /api/query — answer a question grounded in the knowledge graph.

Credentials live in the process environment (set via ``POST /api/credentials``);
this route does not accept them.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from graphiti_rag.api.models import QueryRequest, QueryResponse
from graphiti_rag.api.state import CredentialsNotSet, app_state

logger = logging.getLogger(__name__)
router = APIRouter()

PIPELINE_TIMEOUT_SECONDS = 300


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        pipeline = await app_state.get_pipeline()
    except CredentialsNotSet as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    try:
        answer = await asyncio.wait_for(
            pipeline.query(query=request.query),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Query timed out after %ds", PIPELINE_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="Request timed out")
    except ValueError as exc:
        logger.warning("Query rejected: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal server error")
    return QueryResponse(answer=answer)
