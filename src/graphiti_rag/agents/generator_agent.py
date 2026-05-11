"""Generator agent — produces the final natural-language answer.

Calls the configured OpenAI chat model with a system prompt that grounds
it strictly in the graph-derived context. When the context indicates no
results, the model is instructed to say so plainly rather than confabulate.
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

from graphiti_rag.agents.context_agent import EMPTY_CONTEXT
from graphiti_rag.config import Config
from graphiti_rag.tools.retry import retry_async

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a careful question-answering assistant grounded in a knowledge "
    "graph. You will be given a CONTEXT block extracted from the graph and a "
    "user question. Answer using ONLY the information in CONTEXT. If the "
    "CONTEXT does not contain enough information, say so plainly and do not "
    "speculate. Cite specific entity names from the context where useful. "
    "Be concise."
)


class GeneratorAgent:
    """Produces the final answer from the user query and graph-derived context."""

    def __init__(
        self,
        config: Config,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._config = config
        self._client = client if client is not None else AsyncOpenAI(
            api_key=config.openai_api_key
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
                temperature=0.2,
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
