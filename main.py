"""Development entrypoint — thin wiring layer.

Loads :class:`Config` from ``.env``, builds a :class:`Pipeline`, runs the
ingest+query round-trip on whatever CLI arguments are supplied. Production
deployments use ``uvicorn graphiti_rag.api.app:app`` instead.

Usage::

    python main.py "What is in this corpus?" /path/to/file.pdf /path/to/data.json
"""
from __future__ import annotations

import asyncio
import logging
import sys

from graphiti_rag.config import Config
from graphiti_rag.orchestration.pipeline import Pipeline

logger = logging.getLogger(__name__)


async def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(argv) < 2:
        logger.error('Usage: python main.py "<question>" [doc_path ...]')
        return 2

    question = argv[1]
    doc_paths = argv[2:]

    config = Config()
    if not config.openai_api_key or not config.neo4j_uri:
        logger.error("Missing OPENAI_API_KEY or NEO4J_URI — populate .env first.")
        return 2

    pipeline = Pipeline.from_config(config)
    if doc_paths:
        result = await pipeline.ingest(doc_paths=doc_paths)
        logger.info(
            "Ingest result: %d ok, %d failed (%s)",
            result.successes,
            result.failures,
            ", ".join(result.failed_files) or "none",
        )
    answer = await pipeline.query(query=question)
    logger.info("Answer: %s", answer)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
