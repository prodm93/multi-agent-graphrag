"""Retrieval plan emitted by :class:`RetrievalPlannerAgent`.

The planner LLM returns JSON conforming to :class:`RetrievalPlan`. Any
deviation (unknown strategy, missing required params, malformed JSON)
causes the planner caller to fall back to the deterministic edge-first
hybrid + rerank path.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Strategy = Literal["edge_hybrid", "centered_rerank", "entity_lookup"]


class RetrievalPlanParams(BaseModel):
    """Strategy-specific params. Only ``entity_name`` is currently used."""

    entity_name: str | None = Field(default=None, description="Entity name for entity_lookup")


class RetrievalPlan(BaseModel):
    strategy: Strategy = Field(..., description="The retrieval strategy to run")
    params: RetrievalPlanParams = Field(default_factory=RetrievalPlanParams)
    reason: str = Field(default="", description="One-line justification")
