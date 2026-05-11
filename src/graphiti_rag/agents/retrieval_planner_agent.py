"""Retrieval planner — picks a retrieval strategy per query using the playbook.

Single LLM call per query. Returns a :class:`RetrievalPlan` for the
context agent to dispatch on. Any failure (LLM error, malformed JSON,
unknown strategy) raises :class:`PlanningFailed`, which the caller treats
as the signal to fall back to the deterministic default path.
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI
from pydantic import ValidationError

from graphiti_rag.config import Config
from graphiti_rag.schemas.retrieval_plan import RetrievalPlan

logger = logging.getLogger(__name__)


class PlanningFailed(RuntimeError):
    """Raised when the planner can't produce a usable :class:`RetrievalPlan`."""


class RetrievalPlannerAgent:
    def __init__(
        self,
        config: Config,
        playbook: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._config = config
        self._playbook = playbook
        self._client = client if client is not None else AsyncOpenAI(
            api_key=config.openai_api_key
        )

    async def plan(self, query: str) -> RetrievalPlan:
        if not query.strip():
            raise PlanningFailed("empty query")

        system = (
            "You are a retrieval strategy router for a knowledge-graph QA system. "
            "Pick the best strategy from the playbook for the user's query. "
            "Return ONLY the JSON object specified at the end of the playbook."
        )
        user = f"{self._playbook}\n\nUser query: {query}"

        try:
            completion = await self._client.chat.completions.create(
                model=self._config.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise PlanningFailed(f"LLM call failed: {exc}") from exc

        content = completion.choices[0].message.content or ""
        try:
            plan = RetrievalPlan.model_validate_json(content)
        except ValidationError as exc:
            raise PlanningFailed(f"plan JSON invalid: {exc}") from exc

        if plan.strategy == "entity_lookup" and not (plan.params.entity_name or "").strip():
            raise PlanningFailed("entity_lookup chosen but entity_name missing")

        logger.info(
            "RetrievalPlannerAgent: chose %s — %s",
            plan.strategy,
            plan.reason or "(no reason given)",
        )
        return plan
