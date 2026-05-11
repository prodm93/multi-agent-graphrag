"""Domain types shared across the pipeline.

Thin frozen dataclasses with slots so tools, agents, and pipeline state
share a single typed contract instead of passing untyped ``dict[str, Any]``
blobs around. ``slots=True`` saves memory and prevents accidental attribute
typos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Node:
    """A graph node returned to the agent layer."""

    id: str
    labels: tuple[str, ...] = ()
    name: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Edge:
    """A graph edge returned to the agent layer."""

    id: str
    edge_type: str
    source_id: str
    target_id: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of an ingestion run.

    `failed_files` is a tuple (immutable, hashable) of basenames of the
    documents that failed. The pipeline counts successes and failures; the
    HTTP layer translates basenames back to original client-supplied
    filenames before responding.
    """

    successes: int
    failures: int
    failed_files: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def all_failed(self) -> bool:
        return self.total > 0 and self.successes == 0

    @property
    def partially_failed(self) -> bool:
        return self.successes > 0 and self.failures > 0
