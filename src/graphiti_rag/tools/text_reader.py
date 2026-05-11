"""Direct UTF-8 text reader used by the graph agent for `.txt` inputs.

Graphiti natively ingests free-form text via ``EpisodeType.text`` — there
is no structural extraction to perform. Like :class:`JsonReader`, this is
deliberately **not** a ``DocumentLoader`` subclass and is **not**
registered in ``DocumentLoader._LOADERS`` so the native-text path stays
separate from formats that genuinely need parsing.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TextReader:
    """Reads a UTF-8 ``.txt`` file as-is and returns its contents."""

    def read_sync(self, path: str) -> str:
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"Text file not found: {path}")
        return p.read_text(encoding="utf-8", errors="replace")

    async def read(self, path: str) -> str:
        return await asyncio.to_thread(self.read_sync, path)
