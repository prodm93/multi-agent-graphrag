"""Direct JSON reader used by ontology/graph agents.

JSON inputs must not be flattened to text — Graphiti ingests JSON natively
via ``EpisodeType.json``. This helper is intentionally **not** a
``DocumentLoader`` subclass and is **not** registered in
``DocumentLoader._LOADERS`` so the structured path stays separate.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonReader:
    """Reads ``.json`` files into Python objects without flattening to text."""

    def read_text_sync(self, path: str) -> str:
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"JSON not found: {path}")
        return p.read_text(encoding="utf-8", errors="replace")

    def parse_sync(self, path: str) -> Any:
        return json.loads(self.read_text_sync(path))

    async def read_text(self, path: str) -> str:
        return await asyncio.to_thread(self.read_text_sync, path)

    async def parse(self, path: str) -> Any:
        return await asyncio.to_thread(self.parse_sync, path)
