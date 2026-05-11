"""Pydantic request and response models for the HTTP API.

Credentials are not part of these models — the backend pulls them from
``os.environ`` (set via ``POST /api/credentials``), so the request bodies
only carry payload.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

QUERY_MAX = 2000


class QueryRequest(BaseModel):
    """Body of POST /api/query."""

    query: str = Field(
        ...,
        max_length=QUERY_MAX,
        description="Natural-language question for the graph",
    )


class QueryResponse(BaseModel):
    """Body of the response from POST /api/query."""

    answer: str = Field(..., description="Graph-grounded answer to the query")


class IngestResponse(BaseModel):
    """Body of the response from POST /api/ingest."""

    message: str = Field(..., description="Human-readable summary of the outcome")
    files_received: int = Field(..., description="Number of files received in the request")
    successes: int = Field(..., description="Number of files successfully ingested")
    failures: int = Field(..., description="Number of files that failed to ingest")
    failed_files: list[str] = Field(
        default_factory=list,
        description="Original filenames (as supplied by the client) that failed to ingest",
    )
