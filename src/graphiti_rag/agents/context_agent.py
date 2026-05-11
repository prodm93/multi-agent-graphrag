"""Context agent — formulates LLM-ready context from the knowledge graph.

Edge-first retrieval: Graphiti's hybrid search ranks fact edges by
relevance to the query (BM25 + cosine over fact text, fused via RRF), then
a second pass reranks by graph distance from the top hit
(``EDGE_HYBRID_SEARCH_NODE_DISTANCE``). Endpoint UUIDs are resolved to
entity names so the generator sees readable subjects/objects, and source
episode chunks are pulled per surfaced edge so the generator can ground
specific phrasing back to the corpus.
"""
from __future__ import annotations

import logging

from graphiti_rag.agents.retrieval_planner_agent import RetrievalPlannerAgent
from graphiti_rag.config import Config
from graphiti_rag.domain import Edge
from graphiti_rag.schemas.retrieval_plan import RetrievalPlan
from graphiti_rag.tools.graph_tools import GraphTools

logger = logging.getLogger(__name__)

EMPTY_CONTEXT = "No relevant information found in the knowledge graph."

EDGE_LIMIT = 20
MAX_CHUNKS = 8
MAX_CHUNK_CHARS = 800
MAX_CONTEXT_CHARS = 24000


class ContextAgent:
    """Builds a structured context string for the generator from the graph."""

    def __init__(
        self,
        config: Config,
        graph_tools: GraphTools,
        planner_agent: RetrievalPlannerAgent | None = None,
    ) -> None:
        self._config = config
        self._graph_tools = graph_tools
        self._planner_agent = planner_agent

    async def run(self, query: str) -> str:
        if not query.strip():
            return EMPTY_CONTEXT
        if self._planner_agent is None:
            return await self._run_centered_rerank(query)
        try:
            plan = await self._planner_agent.plan(query)
            rendered = await self._run_plan(query, plan)
            if rendered != EMPTY_CONTEXT:
                return rendered
            logger.warning("Planned retrieval returned empty context; falling back")
        except Exception:
            logger.exception("Planned retrieval failed; falling back")
        return await self._run_centered_rerank(query)

    async def _run_plan(self, query: str, plan: RetrievalPlan) -> str:
        if plan.strategy == "edge_hybrid":
            return await self._run_edge_hybrid(query)
        if plan.strategy == "centered_rerank":
            return await self._run_centered_rerank(query)
        if plan.strategy == "entity_lookup":
            return await self._run_entity_lookup(plan.params.entity_name or "")
        raise ValueError(f"Unsupported retrieval strategy: {plan.strategy}")

    async def _run_edge_hybrid(self, query: str) -> str:
        group_id = self._config.graphiti_namespace
        edges = await self._graph_tools.search_edges(
            query=query,
            group_id=group_id,
            limit=EDGE_LIMIT,
        )
        return await self._render_edges(edges, group_id)

    async def _run_centered_rerank(self, query: str) -> str:
        group_id = self._config.graphiti_namespace
        edges = await self._graph_tools.search_edges(
            query=query,
            group_id=group_id,
            limit=EDGE_LIMIT,
        )
        if not edges:
            return EMPTY_CONTEXT

        anchor = edges[0].source_id
        reranked = await self._graph_tools.search_edges(
            query=query,
            group_id=group_id,
            limit=EDGE_LIMIT,
            center_node_uuid=anchor,
        )
        return await self._render_edges(reranked or edges, group_id)

    async def _run_entity_lookup(self, entity_name: str) -> str:
        if not entity_name.strip():
            return EMPTY_CONTEXT
        group_id = self._config.graphiti_namespace
        nodes = await self._graph_tools.find_entity_by_name(
            entity_name=entity_name,
            group_id=group_id,
            limit=1,
        )
        if not nodes:
            return EMPTY_CONTEXT
        edges = await self._graph_tools.fetch_edges(
            node_id=nodes[0].id,
            group_id=group_id,
            limit=EDGE_LIMIT,
        )
        return await self._render_edges(edges, group_id)

    async def _render_edges(self, edges: list[Edge], group_id: str) -> str:
        if not edges:
            return EMPTY_CONTEXT

        endpoint_uuids = list(
            {e.source_id for e in edges} | {e.target_id for e in edges}
        )
        name_by_uuid = await self._graph_tools.fetch_node_names(
            uuids=endpoint_uuids,
            group_id=group_id,
        )

        episode_order: list[str] = []
        seen_episodes: set[str] = set()
        for edge in edges:
            for ep_uuid in edge.properties.get("episodes", []) or []:
                if ep_uuid not in seen_episodes:
                    seen_episodes.add(ep_uuid)
                    episode_order.append(ep_uuid)
                    if len(episode_order) >= MAX_CHUNKS:
                        break
            if len(episode_order) >= MAX_CHUNKS:
                break

        chunks = await self._graph_tools.fetch_episode_chunks(
            episode_uuids=episode_order,
            group_id=group_id,
        )

        rendered = self._render(edges, name_by_uuid, episode_order, chunks)
        if len(rendered) > MAX_CONTEXT_CHARS:
            rendered = rendered[:MAX_CONTEXT_CHARS].rstrip() + "\n…"
        return rendered

    @staticmethod
    def _render(
        edges: list[Edge],
        name_by_uuid: dict[str, str],
        episode_order: list[str],
        chunks: dict[str, dict[str, str]],
    ) -> str:
        def label(uuid: str) -> str:
            return name_by_uuid.get(uuid) or uuid[:8]

        facts_block: list[str] = ["## Most relevant facts"]
        for edge in edges:
            src = label(edge.source_id)
            tgt = label(edge.target_id)
            props = edge.properties or {}
            fact = props.get("fact", "")
            line = f"- **{edge.edge_type}** ({src} → {tgt})"
            if fact:
                line += f": {fact}"
            time_bits: list[str] = []
            valid_at = props.get("valid_at")
            invalid_at = props.get("invalid_at")
            ref = props.get("reference_time")
            if valid_at:
                time_bits.append(f"valid from {valid_at}")
            if invalid_at:
                time_bits.append(f"invalid from {invalid_at}")
            if ref and not valid_at:
                time_bits.append(f"observed at {ref}")
            if time_bits:
                line += f"  _[{'; '.join(time_bits)}]_"
            facts_block.append(line)
        sections: list[str] = ["\n".join(facts_block)]

        if episode_order and chunks:
            excerpt_block: list[str] = ["## Source excerpts"]
            for ep_uuid in episode_order:
                chunk = chunks.get(ep_uuid)
                if not chunk:
                    continue
                content = chunk.get("content", "").strip()
                if not content:
                    continue
                if len(content) > MAX_CHUNK_CHARS:
                    content = content[:MAX_CHUNK_CHARS].rstrip() + "…"
                name = chunk.get("name", "") or ep_uuid[:8]
                excerpt_block.append(f"### {name}\n{content}")
            if len(excerpt_block) > 1:
                sections.append("\n\n".join(excerpt_block))

        return (
            "\n\n".join(sections)
            if sections[0] != "## Most relevant facts" or len(facts_block) > 1
            else EMPTY_CONTEXT
        )