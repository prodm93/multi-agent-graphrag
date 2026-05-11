"""PDF loader (pdfplumber)."""
from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


class PdfLoader:
    """Extracts text from a PDF file, page by page."""

    def load(self, path: str) -> str:
        if not Path(path).is_file():
            raise ValueError(f"PDF not found: {path}")
        pages: list[str] = []
        try:
            with pdfplumber.open(path) as pdf:
                for index, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        # Single corrupt page: keep going, log once.
                        logger.warning(
                            "PdfLoader: skipping unreadable page %d in %s", index, path
                        )
                        text = ""
                    if text:
                        pages.append(text)
        except Exception as exc:
            logger.warning("PdfLoader: best-effort extraction for %s (%s)", path, exc)
        return "\n\n".join(pages)
