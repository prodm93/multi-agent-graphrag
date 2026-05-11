"""Graph agent — populates the knowledge graph by calling Graphiti.add_episode.

Per-document dispatch:

* ``.json`` → read file content, validate it parses to a non-empty
  ``dict`` / ``list`` via ``json.loads``, then pass the original JSON text
  with ``source=EpisodeType.json``. Empty (``{}``, ``[]``, ``null``,
  whitespace) or invalid JSON files are counted as failures.
* ``.txt`` → read the file as UTF-8 and pass directly with
  ``source=EpisodeType.text``; Graphiti ingests free-form text natively,
  so no parser is involved. Whitespace-only files are counted as failures.
* anything else → ``await self._loader.load(path)`` to extract text, pass
  with ``source=EpisodeType.text``. Whitespace-only extractions are
  counted as failures.

Every call includes ``group_id=self._config.graphiti_namespace`` so each
session's graph is isolated. The ontology is compiled into Graphiti's
``entity_types``/``edge_types``/``edge_type_map`` keyword arguments via
:func:`compile_ontology`.

Per-document failures are logged and skipped — one bad PDF must not abort
ingestion of the rest of the batch. The aggregate outcome is returned as
an :class:`IngestResult`.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core.nodes import EpisodeType

from graphiti_rag.config import Config
from graphiti_rag.domain import IngestResult
from graphiti_rag.graph.graphiti_client import GraphitiClient
from graphiti_rag.schemas.ontology import OntologyDefinition
from graphiti_rag.schemas.ontology_compiler import compile_ontology
from graphiti_rag.tools.document_loader import DocumentLoader
from graphiti_rag.tools.json_reader import JsonReader
from graphiti_rag.tools.text_reader import TextReader
from graphiti_rag.tools.retry import retry_async

logger = logging.getLogger(__name__)


class GraphAgent:
    """Drives Graphiti to ingest documents and materialise the knowledge graph."""

    def __init__(
        self,
        config: Config,
        graphiti: GraphitiClient,
        loader: DocumentLoader,
    ) -> None:
        self._config = config
        self._graphiti = graphiti
        self._loader = loader
        self._json_reader = JsonReader()
        self._text_reader = TextReader()

    async def run(
        self,
        doc_paths: list[str],
        ontology: OntologyDefinition,
    ) -> IngestResult:
        if not doc_paths:
            logger.info("GraphAgent: no documents to ingest")
            return IngestResult(successes=0, failures=0)

        entity_types, edge_types, edge_type_map = compile_ontology(ontology)
        group_id = self._config.graphiti_namespace
        logger.info(
            "GraphAgent: ingesting %d documents into group=%s "
            "(ontology: %d entity types, %d edge types, %d mappings)",
            len(doc_paths),
            group_id,
            len(entity_types),
            len(edge_types),
            len(edge_type_map),
        )

        successes = 0
        failures = 0
        failed: list[str] = []
        for path in doc_paths:
            try:
                await self._ingest_one(
                    path=path,
                    group_id=group_id,
                    entity_types=entity_types,
                    edge_types=edge_types,
                    edge_type_map=edge_type_map,
                )
                successes += 1
            except Exception:
                failures += 1
                failed.append(Path(path).name)
                logger.exception("GraphAgent: failed to ingest %s", path)

        logger.info(
            "GraphAgent: ingestion complete — %d ok, %d failed", successes, failures
        )
        return IngestResult(
            successes=successes,
            failures=failures,
            failed_files=tuple(failed),
        )

    async def _ingest_one(
        self,
        *,
        path: str,
        group_id: str,
        entity_types: dict,
        edge_types: dict,
        edge_type_map: dict,
    ) -> None:
        suffix = Path(path).suffix.lower()
        name = Path(path).name
        if suffix == ".json":
            episode_body = await self._json_reader.read_text(path)
            if not episode_body.strip():
                raise ValueError(f"JSON document is empty: {name}")
            try:
                parsed = json.loads(episode_body)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON document is invalid: {name}") from exc
            if parsed is None or parsed == {} or parsed == []:
                raise ValueError(f"JSON document has no content: {name}")
            source = EpisodeType.json
            source_description = f"JSON document: {name}"
        elif suffix == ".txt":
            episode_body = await self._text_reader.read(path)
            if not episode_body.strip():
                raise ValueError(f"Text document is empty: {name}")
            source = EpisodeType.text
            source_description = f"Text document: {name}"
        else:
            episode_body = await self._loader.load(path)
            if not episode_body.strip():
                raise ValueError(f"Document yielded no extractable text: {name}")
            source = EpisodeType.text
            source_description = f"Document: {name}"

        async def _call_add_episode() -> None:
            await self._graphiti.graphiti.add_episode(
                name=name,
                episode_body=episode_body,
                source_description=source_description,
                reference_time=datetime.now(timezone.utc),
                source=source,
                group_id=group_id,
                entity_types=entity_types or None,
                edge_types=edge_types or None,
                edge_type_map=edge_type_map or None,
            )

        await retry_async(
            _call_add_episode,
            attempts=3,
            base_delay=2.0,
            max_delay=15.0,
            label=f"add_episode[{name}]",
        )
