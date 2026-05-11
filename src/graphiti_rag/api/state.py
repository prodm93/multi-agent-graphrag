"""Process-wide singleton holding the live :class:`Pipeline`.

The pipeline is built lazily from the current process environment (which
mirrors ``.env``). Submitting credentials via ``POST /api/credentials``
rewrites ``.env``, reloads it into ``os.environ``, and calls
:meth:`AppState.reset` so the next ingest/query call rebuilds.
"""
from __future__ import annotations

import asyncio
import logging

from graphiti_rag.config import Config
from graphiti_rag.orchestration.pipeline import Pipeline

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._lock = asyncio.Lock()

    async def get_pipeline(self) -> Pipeline:
        async with self._lock:
            if self._pipeline is None:
                config = Config()
                if not config.openai_api_key:
                    raise CredentialsNotSet("OpenAI API key is not set")
                if not config.neo4j_uri:
                    raise CredentialsNotSet("Neo4j URI is not set")
                logger.info("AppState: building pipeline from current env")
                self._pipeline = Pipeline.from_config(config)
            return self._pipeline

    async def reset(self) -> None:
        async with self._lock:
            if self._pipeline is not None:
                logger.info("AppState: clearing pipeline so next call rebuilds")
                self._pipeline = None


class CredentialsNotSet(RuntimeError):
    """Raised by :meth:`AppState.get_pipeline` when required env vars are absent."""


app_state = AppState()
