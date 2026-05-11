"""Resolve LLM-emitted type names to concrete Python types.

The LLM produces string type annotations like ``"str"``, ``"int | None"``,
``"list[str]"``. We never feed those to ``eval`` — instead, an explicit
allow-list maps each accepted string to a real type. Anything outside the
allow-list collapses to :class:`str`, which is the safe default for free-form
LLM output.
"""
from __future__ import annotations

from typing import Any

BASE_TYPES: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list[str]": list[str],
    "list[int]": list[int],
    "list[float]": list[float],
    "dict[str, str]": dict[str, str],
    "dict[str, Any]": dict[str, Any],
}


def resolve_type(type_name: str) -> Any:
    """Return the Python type for ``type_name``.

    Handles ``"X | None"`` / ``"X| None"`` suffixes by wrapping the resolved
    base type in ``Optional``. Unknown base types fall back to :class:`str`.
    """
    raw = type_name.strip()
    if " | None" in raw or raw.endswith("| None"):
        base = raw.replace("| None", "").strip().rstrip("|").strip()
        return BASE_TYPES.get(base, str) | None
    return BASE_TYPES.get(raw, str)
