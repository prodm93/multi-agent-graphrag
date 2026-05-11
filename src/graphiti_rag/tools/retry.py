"""Async retry helper with exponential backoff and jitter."""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    label: str = "operation",
) -> T:
    """Call an async ``fn`` with exponential backoff + jitter.

    Re-raises the last exception if all attempts fail. Designed for
    transient failures (network blips, rate limits) — caller should not
    pass functions whose errors are deterministic.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                logger.warning(
                    "%s: failed after %d attempts (%s)", label, attempts, exc
                )
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)
            logger.info(
                "%s: attempt %d/%d failed (%s); retrying in %.2fs",
                label,
                attempt,
                attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    # Unreachable, but keep type checker happy.
    assert last_exc is not None
    raise last_exc
