"""Dispatcher that routes documents to the appropriate format-specific loader.

Only formats that Graphiti cannot ingest natively are listed here. JSON is
omitted on purpose — `GraphAgent` reads `.json` files directly and feeds them
to `add_episode(..., source=EpisodeType.json)`, preserving structure for
Graphiti's native entity/relationship extraction.

The public interface is async-first: callers ``await loader.load(path)``.
The synchronous variant ``_load_sync`` is private and exists only so the
async wrapper can dispatch the underlying CPU-bound parsers via
``asyncio.to_thread`` without blocking the event loop.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from graphiti_rag.tools.loaders.csv_loader import CsvLoader
from graphiti_rag.tools.loaders.docx_loader import DocxLoader
from graphiti_rag.tools.loaders.pdf_loader import PdfLoader
from graphiti_rag.tools.loaders.xlsx_loader import XlsxLoader

logger = logging.getLogger(__name__)

_LOADERS: dict[str, type] = {
    ".pdf": PdfLoader,
    ".csv": CsvLoader,
    ".docx": DocxLoader,
    ".xlsx": XlsxLoader,
}


class DocumentLoader:
    """Dispatches document loading to the appropriate format-specific loader."""

    async def load(self, path: str) -> str:
        """Load a document and return its text content.

        The underlying parsing libraries (pdfplumber, python-docx, openpyxl)
        are sync and CPU-bound; calling them from the event loop directly
        would block. ``asyncio.to_thread`` runs the sync dispatch on the
        default executor.
        """
        return await asyncio.to_thread(self._load_sync, path)

    def _load_sync(self, path: str) -> str:
        """Synchronous loader dispatch — only called by ``load`` via to_thread."""
        suffix = Path(path).suffix.lower()
        loader_cls = _LOADERS.get(suffix)
        if loader_cls is None:
            raise ValueError(f"Unsupported document format: {suffix}")
        logger.info("Loading %s with %s", path, loader_cls.__name__)
        return loader_cls().load(path)
