"""Generator agent — produces the final natural-language answer.

Calls the configured OpenAI chat model with a system prompt that grounds
it strictly in the graph-derived context. When the context indicates no
results, the model is instructed to say so plainly rather than confabulate.
"""
from __future__ import annotations

import logging

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from graphiti_rag.agents.context_agent import EMPTY_CONTEXT
from graphiti_rag.config import Config
from graphiti_rag.tools.retry import retry_async

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an expert analyst answering questions strictly grounded in a "
    "knowledge graph. You will receive a CONTEXT block extracted from the "
    "graph and a user QUESTION. Answer using ONLY information that appears "
    "in CONTEXT.\n"
    "\n"
    "Reasoning protocol (silent — do NOT emit it):\n"
    "  1. Scan the CONTEXT facts and identify which entities and edges are "
    "actually relevant to the question.\n"
    "  2. Assemble the answer from those edges. Note any gaps before you "
    "commit.\n"
    "  3. If a key piece is missing, name what's missing rather than "
    "inferring.\n"
    "\n"
    "Citation discipline: every non-trivial factual claim must reference the "
    "edge it came from, formatted exactly as `[SOURCE_ENTITY → EDGE_TYPE → "
    "TARGET_ENTITY]` using the entity names and edge type as they appear in "
    "the CONTEXT. Only entities present in CONTEXT may be cited. Do not "
    "invent entities, edge types, or relationships.\n"
    "\n"
    "Answer format by question type:\n"
    "  • Yes/no: lead with the verdict, then one supporting sentence with a "
    "citation.\n"
    "  • Factual lookup: 1-2 sentences, with citations.\n"
    "  • Analytical or comparative: a brief opening sentence then bullet "
    "points (each bullet cites its supporting edge).\n"
    "  • List request: just the list, one citation per item.\n"
    "\n"
    "Missing-information protocol: if CONTEXT does not contain enough to "
    "answer, say so plainly and name the specific piece that is missing "
    "(e.g. \"the context shows X was approved but does not state the "
    "approval date\"). Do not speculate. Do not fall back on general "
    "knowledge.\n"
    "\n"
    "Temporal handling: where an edge carries `valid from` / `invalid from` / "
    "`observed at` bounds in CONTEXT, respect those bounds in the answer. If "
    "the question is about the present and an edge is marked invalid, say "
    "so. Surface temporal bounds when they are material to the answer.\n"
    "\n"
    "Length: 4-10 sentences (or bullets) for analytical questions; 1-2 "
    "sentences for factual lookups; bullets only for analytical, comparative, "
    "or list requests."
)


class GeneratorAgent:
    """Produces the final answer from the user query and graph-derived context."""

    def __init__(
        self,
        config: Config,
        client: AsyncOpenAI | None = None,
        langsmith_client: object | None = None,
    ) -> None:
        self._config = config
        # ``wrap_openai`` is a passthrough when LangSmith tracing is off, so
        # wrap unconditionally; env-var-driven instrumentation decides
        # whether spans are actually emitted. ``langsmith_client``, when
        # supplied, carries the per-tier privacy filters (anonymiser, or
        # hide_inputs/hide_outputs) derived from the user's consent choice.
        if client is not None:
            self._client = client
        else:
            tracing_extra = (
                {"client": langsmith_client}
                if langsmith_client is not None
                else None
            )
            self._client = wrap_openai(
                AsyncOpenAI(api_key=config.openai_api_key),
                tracing_extra=tracing_extra,
            )

    async def run(self, query: str, context: str) -> str:
        if context.strip() == EMPTY_CONTEXT:
            return (
                "I could not find anything relevant to your question in the "
                "knowledge graph yet. Try ingesting more documents or "
                "rephrasing the question."
            )

        async def _call() -> str:
            completion = await self._client.chat.completions.create(
                model=self._config.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}",
                    },
                ],
                temperature=0.0,
            )
            return (completion.choices[0].message.content or "").strip()

        try:
            answer = await retry_async(
                _call,
                attempts=3,
                base_delay=1.0,
                max_delay=8.0,
                label="GeneratorAgent.chat",
            )
        except Exception:
            logger.exception("GeneratorAgent: chat completion failed")
            return (
                "An error occurred while generating the answer. Please retry "
                "in a moment."
            )
        return answer
