"""Process-wide singleton holding the live :class:`Pipeline`.

The pipeline is built lazily from the current process environment (which
mirrors ``.env``). Submitting credentials via ``POST /api/credentials``
rewrites ``.env``, reloads it into ``os.environ``, and calls
:meth:`AppState.reset` so the next ingest/query call rebuilds.

The privacy consent tier (from the modal) is also tracked here. Changing
the tier resets the pipeline so the next call rebuilds with a LangSmith
client configured for the new tier — though when no ``LANGSMITH_API_KEY``
is in the environment (the default for clone-and-run), this wiring is
inert because ``wrap_openai`` becomes a no-op.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from graphiti_rag.config import Config
from graphiti_rag.observability import ConsentTier, build_langsmith_client
from graphiti_rag.orchestration.pipeline import Pipeline

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._consent_tier: Optional[ConsentTier] = None
        self._lock = asyncio.Lock()

    async def get_pipeline(self) -> Pipeline:
        async with self._lock:
            if self._pipeline is None:
                config = Config()
                if not config.openai_api_key:
                    raise CredentialsNotSet("OpenAI API key is not set")
                if not config.neo4j_uri:
                    raise CredentialsNotSet("Neo4j URI is not set")
                logger.info(
                    "AppState: building pipeline from current env (consent_tier=%s)",
                    self._consent_tier,
                )
                langsmith_client = build_langsmith_client(self._consent_tier)
                self._pipeline = Pipeline.from_config(
                    config, langsmith_client=langsmith_client
                )
            return self._pipeline

    async def set_consent_tier(self, tier: ConsentTier) -> None:
        """Record the user's consent choice and clear the cached pipeline.

        The next ingest/query call rebuilds the pipeline so the new tier's
        privacy filters take effect on subsequently emitted LangSmith traces.
        """
        async with self._lock:
            if self._consent_tier == tier and self._pipeline is None:
                return
            self._consent_tier = tier
            if self._pipeline is not None:
                logger.info(
                    "AppState: consent_tier=%s; clearing pipeline so next call rebuilds",
                    tier,
                )
                self._pipeline = None

    async def reset(self) -> None:
        async with self._lock:
            if self._pipeline is not None:
                logger.info("AppState: clearing pipeline so next call rebuilds")
                self._pipeline = None


class CredentialsNotSet(RuntimeError):
    """Raised by :meth:`AppState.get_pipeline` when required env vars are absent."""


app_state = AppState()
