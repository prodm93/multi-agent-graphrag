"""Ontology agent — infers a serialisable ontology from a sample of the corpus.

Samples up to 5 documents (truncated to keep prompts cheap), asks the LLM
to emit an :class:`OntologyDefinition` as JSON conforming to its
``model_json_schema``, and returns the parsed object. JSON inputs are read
directly via :class:`JsonReader` so structure is preserved during sampling.

Fallback semantics
------------------
An empty :class:`OntologyDefinition` returned by :meth:`run` means
**"use Graphiti's default extraction with no custom ontology"**, *not*
"successfully inferred an ontology that happens to be empty". The fallback
path goes through :meth:`_empty_fallback_ontology` and is always
accompanied by a warning log so operators can spot silent degradations.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import ValidationError

from graphiti_rag.config import Config
from graphiti_rag.schemas.ontology import OntologyDefinition
from graphiti_rag.tools.document_loader import DocumentLoader
from graphiti_rag.tools.json_reader import JsonReader
from graphiti_rag.tools.retry import retry_async
from graphiti_rag.tools.text_reader import TextReader

logger = logging.getLogger(__name__)

MAX_DOCS_SAMPLED = 5
MAX_CHARS_PER_DOC = 2000
MAX_JSON_SAMPLE_BYTES = 4000


class OntologyAgent:
    """Infers a serialisable ontology (entity types, edge types, mappings)."""

    def __init__(
        self,
        config: Config,
        loader: DocumentLoader,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._config = config
        self._loader = loader
        self._json_reader = JsonReader()
        self._text_reader = TextReader()
        self._client = client if client is not None else AsyncOpenAI(
            api_key=config.openai_api_key
        )

    async def run(self, doc_paths: list[str]) -> OntologyDefinition:
        if not doc_paths:
            return self._empty_fallback_ontology(reason="no documents supplied")

        samples = await self._collect_samples(doc_paths[:MAX_DOCS_SAMPLED])
        if not samples:
            return self._empty_fallback_ontology(
                reason=f"no readable samples in {len(doc_paths)} document(s)"
            )

        try:
            ontology = await retry_async(
                lambda: self._infer(samples),
                attempts=3,
                base_delay=1.0,
                max_delay=8.0,
                label="OntologyAgent.infer",
            )
        except Exception:
            logger.exception("OntologyAgent: ontology inference failed")
            return self._empty_fallback_ontology(reason="LLM inference error")
        logger.info(
            "OntologyAgent: inferred %d entity types, %d edge types, %d mappings",
            len(ontology.entity_types),
            len(ontology.edge_types),
            len(ontology.edge_type_map),
        )
        return ontology

    def _empty_fallback_ontology(self, *, reason: str) -> OntologyDefinition:
        """Return an empty ontology and log that ingestion will run un-ontologised.

        An empty :class:`OntologyDefinition` is the *signal* to
        :class:`GraphAgent` that no custom ``entity_types`` / ``edge_types``
        / ``edge_type_map`` should be passed to ``add_episode``; Graphiti
        then falls back to its default extraction behaviour.
        """
        logger.warning(
            "OntologyAgent: returning empty ontology — ingestion will proceed "
            "without custom ontology (Graphiti default extraction). Reason: %s",
            reason,
        )
        return OntologyDefinition()

    async def _collect_samples(self, doc_paths: list[str]) -> list[tuple[str, str]]:
        samples: list[tuple[str, str]] = []
        for path in doc_paths:
            try:
                sample = await self._sample_one(path)
            except Exception:
                logger.warning("OntologyAgent: skipping unreadable %s", path, exc_info=True)
                continue
            if sample:
                samples.append((Path(path).name, sample))
        return samples

    async def _sample_one(self, path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            text = await self._json_reader.read_text(path)
            return text[:MAX_JSON_SAMPLE_BYTES]
        if suffix == ".txt":
            text = await self._text_reader.read(path)
            return text[:MAX_CHARS_PER_DOC]
        text = await self._loader.load(path)
        return text[:MAX_CHARS_PER_DOC]

    async def _infer(self, samples: list[tuple[str, str]]) -> OntologyDefinition:
        schema = json.dumps(OntologyDefinition.model_json_schema(), indent=2)
        sample_blocks = "\n\n".join(
            f"### {name}\n{content}" for name, content in samples
        )
        system = (
            "You are an ontology engineer. From the provided document samples, "
            "infer a knowledge-graph ontology: entity TYPES, edge TYPES, and the "
            "allowed (source_entity_type, target_entity_type) -> [edge_type_names] "
            "mappings. Do NOT extract specific instances — only types and mappings. "
            "Each field must include a concise description. Use Python-style "
            "type names like 'str', 'int', 'float', 'bool', or 'str | None'."
        )
        user = (
            f"Return ONLY a JSON object that validates against this schema:\n"
            f"```json\n{schema}\n```\n\n"
            f"Document samples:\n{sample_blocks}"
        )
        completion = await self._client.chat.completions.create(
            model=self._config.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        try:
            return OntologyDefinition.model_validate_json(content)
        except ValidationError as exc:
            logger.warning("OntologyAgent: validation failed (%s)", exc)
            raise
