"""Loader for the retrieval-strategy playbook.

The playbook text lives as a packaged asset
(``graphiti_rag/assets/retrieval_playbook.md``) and is loaded once at
import time. Keeping the prose out of Python lets it grow without
bloating the module and lets non-Python contributors edit it as plain
Markdown.
"""
from __future__ import annotations

from importlib.resources import files

PLAYBOOK: str = (
    files("graphiti_rag.assets")
    .joinpath("retrieval_playbook.md")
    .read_text(encoding="utf-8")
)
