"""Retrieval planner — picks a retrieval strategy per query using the playbook.

Single LLM call per query. Returns a :class:`RetrievalPlan` for the
context agent to dispatch on. Any failure (LLM error, malformed JSON,
unknown strategy) raises :class:`PlanningFailed`, which the caller treats
as the signal to fall back to the deterministic default path.
"""
from __future__ import annotations

import logging

from langsmith.wrappers import wrap_openai
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
        langsmith_client: object | None = None,
    ) -> None:
        self._config = config
        self._playbook = playbook
        # Constructed on first .plan() call when not supplied — keeps any
        # credential problems out of agent construction so the caller's
        # fallback path can absorb them uniformly with other LLM errors.
        self._client = client
        # When supplied, this LangSmith client carries the privacy filters
        # derived from the user's consent tier (anonymiser callable, or
        # hide_inputs/hide_outputs flags). ``None`` means "use wrap_openai's
        # env-driven default", which is a no-op when tracing is off.
        self._langsmith_client = langsmith_client

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            # ``wrap_openai`` is a passthrough when LangSmith tracing is off,
            # so wrap unconditionally and let env-var-driven instrumentation
            # decide whether spans are actually emitted.
            tracing_extra = (
                {"client": self._langsmith_client}
                if self._langsmith_client is not None
                else None
            )
            self._client = wrap_openai(
                AsyncOpenAI(api_key=self._config.openai_api_key),
                tracing_extra=tracing_extra,
            )
        return self._client

    async def plan(self, query: str) -> RetrievalPlan:
        if not query.strip():
            raise PlanningFailed("empty query")

        system = (
            "You are a senior knowledge-graph retrieval specialist. Your single "
            "job is to route a user query to the right retrieval strategy given "
            "the playbook below.\n"
            "\n"
            "Reasoning protocol (silent — do NOT emit it):\n"
            "  1. Scan the query for a focal entity (a proper noun naming one "
            "graph node, not a category).\n"
            "  2. Classify intent: properties-of-X, neighbourhood-of-X, "
            "facts-about-topic, or verification.\n"
            "  3. Check the anti-signals for your first-guess strategy; if any "
            "fire, reconsider.\n"
            "  4. Commit to one strategy.\n"
            "\n"
            "Output discipline: emit ONLY the JSON object specified at the end "
            "of the playbook. No prose, no markdown code fences, no "
            "explanation outside the JSON.\n"
            "\n"
            "`reason` field: one short sentence that references a SPECIFIC "
            "signal from the playbook (e.g. 'properties-of-X over a single "
            "named subject', 'multi-entity comparison'). Do NOT restate the "
            "query. Do NOT summarise the playbook generically.\n"
            "\n"
            "If you are genuinely uncertain between two strategies, prefer "
            "`edge_hybrid` — it is the safest general-purpose default and "
            "matches the deterministic fallback path. For non-empty but "
            "essentially unparseable queries, still emit `edge_hybrid` with a "
            "`reason` that names the unparseability."
        )
        user = f"{self._playbook}\n\nUser query: {query}"

        try:
            client = self._ensure_client()
            completion = await client.chat.completions.create(
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
