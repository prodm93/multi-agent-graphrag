"""Sanitise LLM-emitted names into safe Python identifiers.

The LLM is free to emit names like ``"Drug Trial"``, ``"e-mail"``, or
``"123Foo"`` — none of which are valid Python identifiers and therefore
cannot be passed to :func:`pydantic.create_model`. This module produces:

* PascalCase class identifiers via :func:`sanitise_class_name`
* snake_case field identifiers via :func:`sanitise_field_name`

Either function may return ``None`` to signal "drop this name", which the
compiler logs and skips. Originals are preserved separately (in model
docstrings and Field descriptions) so prompt context isn't lost.
"""
from __future__ import annotations

import keyword
import re

_NON_WORD = re.compile(r"[^0-9A-Za-z]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _strip_to_words(name: str) -> list[str]:
    """Split ``name`` into bare alphanumeric word tokens."""
    if not name:
        return []
    spaced = _CAMEL_BOUNDARY.sub(" ", name)
    parts = _NON_WORD.split(spaced)
    return [p for p in parts if p]


def sanitise_class_name(raw: str) -> str | None:
    """Return a safe PascalCase class identifier, or ``None`` to drop."""
    parts = _strip_to_words(raw)
    if not parts:
        return None
    cleaned = "".join(p[:1].upper() + p[1:].lower() for p in parts)
    if not cleaned or not cleaned[0].isalpha():
        # Class names must start with a letter (Pydantic / Python rule).
        cleaned = f"T{cleaned}" if cleaned else None
    return cleaned


def sanitise_field_name(raw: str) -> str | None:
    """Return a safe snake_case field identifier, or ``None`` to drop."""
    parts = _strip_to_words(raw)
    if not parts:
        return None
    cleaned = "_".join(p.lower() for p in parts)
    if not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = f"f_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned
