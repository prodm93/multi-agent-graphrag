"""CSV loader (stdlib csv)."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CsvLoader:
    """Reads a CSV and renders rows as a single readable string."""

    def load(self, path: str) -> str:
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"CSV not found: {path}")
        rows: list[str] = []
        try:
            with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.reader(fh)
                header: list[str] | None = None
                for index, row in enumerate(reader):
                    if index == 0:
                        header = row
                        rows.append(" | ".join(row))
                        continue
                    if header is not None and len(row) == len(header):
                        rows.append(
                            ", ".join(f"{h}: {v}" for h, v in zip(header, row) if v)
                        )
                    else:
                        rows.append(" | ".join(row))
        except Exception as exc:
            logger.warning("CsvLoader: best-effort extraction for %s (%s)", path, exc)
        return "\n".join(rows)
