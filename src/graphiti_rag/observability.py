"""LangSmith client construction per privacy consent tier.

Three tiers represent the user-facing data-sharing choice surfaced by the
privacy modal:

* ``full``           — payloads sent as-is to LangSmith.
* ``anonymised``     — payloads sent with a regex-based PII scrub applied
                       (emails, phone numbers, long alphanumeric tokens,
                       obvious API-key shapes).
* ``metadata_only``  — payloads suppressed; only spans, timings, and run
                       structure are recorded.

When ``LANGSMITH_TRACING`` is off (the default in clone-and-run, since the
startup probe in ``api/app.py`` forces it off when no key is present in the
process environment), the ``wrap_openai`` call elsewhere becomes a no-op
regardless of which client is passed in. The tier wiring is therefore inert
for end users but fully exercised — ready to become load-bearing the moment
a hosted deployment supplies a real LangSmith key.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


ConsentTier = Literal["full", "anonymised", "metadata_only"]


# Coarse and conservative — favours over-scrubbing to under-scrubbing.
# Order matters: longer-looking patterns first so they aren't shadowed by
# shorter ones (e.g. API-key tokens before generic hex tokens).
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:sk|pk|lsv2|lsv1)[_-][A-Za-z0-9_-]{16,}\b"),
        "<api-key>",
    ),
    (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "<email>",
    ),
    (
        re.compile(r"\+?\d[\d\s().-]{8,}\d"),
        "<phone>",
    ),
    (
        re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
        "<token>",
    ),
)


def _scrub(value: Any) -> Any:
    """Recursively apply PII patterns to strings inside arbitrary payloads."""
    if isinstance(value, str):
        out = value
        for pattern, replacement in _PII_PATTERNS:
            out = pattern.sub(replacement, out)
        return out
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _anonymiser(payload: dict) -> dict:
    scrubbed = _scrub(payload)
    return scrubbed if isinstance(scrubbed, dict) else {"value": scrubbed}


def build_langsmith_client(tier: Optional[ConsentTier]) -> Optional[object]:
    """Construct a :class:`langsmith.Client` configured for ``tier``.

    Returns ``None`` when:

    * ``tier`` is ``None`` (no explicit user choice has been recorded), or
    * no ``LANGSMITH_API_KEY`` is set in the process environment.

    Callers should treat ``None`` as "no per-tier customisation needed; let
    ``wrap_openai`` use its env-driven default" (which itself is a no-op
    when ``LANGSMITH_TRACING=false``).
    """
    if tier is None:
        return None

    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if api_key == "":
        return None

    endpoint = (
        os.environ.get("LANGSMITH_ENDPOINT", "").strip()
        or "https://api.smith.langchain.com"
    )

    from langsmith import Client

    kwargs: dict[str, Any] = {"api_key": api_key, "api_url": endpoint}
    if tier == "anonymised":
        kwargs["anonymizer"] = _anonymiser
    elif tier == "metadata_only":
        kwargs["hide_inputs"] = True
        kwargs["hide_outputs"] = True
    # "full" → no privacy filters applied to the client.

    try:
        return Client(**kwargs)
    except Exception:
        logger.warning(
            "Failed to construct LangSmith client for tier=%s; tracing wiring "
            "will fall back to env-driven default",
            tier,
            exc_info=True,
        )
        return None
