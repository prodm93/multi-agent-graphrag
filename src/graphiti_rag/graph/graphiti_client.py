"""Graphiti instance lifecycle wrapper.

Builds a :class:`graphiti_core.Graphiti` over a :class:`Neo4jDriver`, with
**all three** OpenAI components explicitly constructed from the session
config: LLM client, embedder, and cross-encoder/reranker. Graphiti's
``__init__`` falls back to default constructors for any of these three if
they are missing, and each default reads ``OPENAI_API_KEY`` from the
process environment — which is exactly the failure mode we hit before.
Wiring all three by hand makes the credential path explicit and stops
Graphiti from silently spinning up an env-reading client we forgot about.
"""
from __future__ import annotations

import logging
from typing import Self

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient

from graphiti_rag.config import Config
from graphiti_rag.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphitiClient:
    """Owns a Graphiti instance bound to the current session credentials."""

    def __init__(self, config: Config, neo4j: Neo4jClient) -> None:
        self._config = config
        self._neo4j = neo4j
        self._graphiti: Graphiti | None = None

    async def initialise(self) -> None:
        if self._graphiti is not None:
            return

        api_key = self._config.openai_api_key
        if not api_key:
            raise RuntimeError("OpenAI API key is not configured")

        driver = Neo4jDriver(
            uri=self._config.neo4j_uri,
            user=self._config.neo4j_username,
            password=self._config.neo4j_password,
            database=self._config.neo4j_database or "neo4j",
        )
        llm_client = OpenAIClient(
            config=LLMConfig(api_key=api_key, model=self._config.openai_model)
        )
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(api_key=api_key)
        )
        cross_encoder = OpenAIRerankerClient(
            config=LLMConfig(api_key=api_key)
        )

        logger.info(
            "GraphitiClient: initialising in namespace %s",
            self._config.graphiti_namespace,
        )
        self._graphiti = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

    async def ensure_indices_and_constraints(self) -> None:
        if self._graphiti is None:
            raise RuntimeError(
                "GraphitiClient.ensure_indices_and_constraints called before initialise"
            )
        await self._graphiti.build_indices_and_constraints()

    async def close(self) -> None:
        if self._graphiti is None:
            return
        try:
            await self._graphiti.close()
        except Exception:
            logger.exception("GraphitiClient: error closing")
        finally:
            self._graphiti = None

    @property
    def graphiti(self) -> Graphiti:
        if self._graphiti is None:
            raise RuntimeError(
                "GraphitiClient is not initialised — call initialise() first"
            )
        return self._graphiti

    async def __aenter__(self) -> Self:
        await self.initialise()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
