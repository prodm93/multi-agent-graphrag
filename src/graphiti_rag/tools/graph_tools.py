"""I/O tools for querying the Graphiti-backed Neo4j graph.

Pure I/O. No prompts, no LLM calls, no business decisions. Anything that
involves judgement belongs in an agent.

* :meth:`search_nodes` delegates to Graphiti's hybrid node search
  (``search_`` with the ``NODE_HYBRID_SEARCH_RRF`` recipe), which fuses
  BM25, cosine similarity, and BFS results before reranking.
* :meth:`fetch_edges` runs a small Cypher query against the underlying
  Neo4j driver to enumerate every Graphiti ``RELATES_TO`` edge incident on
  a given entity within the requested ``group_id``.

Both methods return typed :class:`Node` / :class:`Edge` dataclasses so
agents never see raw driver records or untyped dicts.
"""
from __future__ import annotations

import logging
from typing import Any

from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

from graphiti_rag.domain import Edge, Node
from graphiti_rag.graph.graphiti_client import GraphitiClient
from graphiti_rag.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


_FETCH_EDGES_CYPHER = """
MATCH (n {uuid: $uuid, group_id: $group_id})-[r:RELATES_TO]-(m)
WHERE r.group_id = $group_id
  AND ($edge_type IS NULL OR r.name = $edge_type)
RETURN
  r.uuid AS uuid,
  r.name AS edge_type,
  startNode(r).uuid AS source_id,
  endNode(r).uuid AS target_id,
  r.fact AS fact,
  r.attributes AS attributes,
  r.episodes AS episodes
LIMIT $limit
"""

_FETCH_EPISODE_CHUNKS_CYPHER = """
MATCH (e:Episodic)
WHERE e.uuid IN $uuids AND e.group_id = $group_id
RETURN e.uuid AS uuid, e.name AS name, e.content AS content
"""

_FETCH_NODE_NAMES_CYPHER = """
MATCH (n:Entity)
WHERE n.uuid IN $uuids AND n.group_id = $group_id
RETURN n.uuid AS uuid, n.name AS name
"""

_FIND_ENTITY_BY_NAME_CYPHER = """
MATCH (n:Entity)
WHERE n.group_id = $group_id
  AND toLower(n.name) CONTAINS toLower($entity_name)
RETURN n.uuid AS uuid, n.name AS name
ORDER BY size(n.name) ASC
LIMIT $limit
"""


class GraphTools:
    """Hybrid I/O tools over Graphiti + Neo4j."""

    def __init__(self, graphiti: GraphitiClient, neo4j: Neo4jClient) -> None:
        self._graphiti = graphiti
        self._neo4j = neo4j

    async def find_entity_by_name(
        self,
        entity_name: str,
        group_id: str,
        limit: int = 5,
    ) -> list[Node]:
        """Find entity nodes by name within ``group_id``."""
        if not entity_name.strip():
            return []
        try:
            async with self._neo4j.driver.session(database=self._neo4j.database) as session:
                records = await session.run(
                    _FIND_ENTITY_BY_NAME_CYPHER,
                    entity_name=entity_name,
                    group_id=group_id,
                    limit=limit,
                )
                rows = [record.data() async for record in records]
        except Exception:
            logger.exception(
                "find_entity_by_name failed for entity=%r group=%s",
                entity_name,
                group_id,
            )
            return []

        return [
            Node(
                id=row["uuid"],
                labels=("Entity",),
                name=row.get("name") or "",
                properties={},
            )
            for row in rows
            if row.get("uuid")
        ]


    async def search_nodes(
        self,
        query: str,
        group_id: str,
        limit: int = 10,
    ) -> list[Node]:
        """Hybrid search for entity nodes within ``group_id``."""
        config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = limit
        try:
            results = await self._graphiti.graphiti.search_(
                query=query,
                config=config,
                group_ids=[group_id],
            )
        except Exception:
            logger.exception("search_nodes failed for query=%r group=%s", query, group_id)
            return []
        nodes: list[Node] = []
        for entity in results.nodes:
            nodes.append(
                Node(
                    id=entity.uuid,
                    labels=tuple(entity.labels or ()),
                    name=entity.name,
                    properties={
                        "summary": getattr(entity, "summary", "") or "",
                        **dict(getattr(entity, "attributes", {}) or {}),
                    },
                )
            )
        return nodes

    async def fetch_edges(
        self,
        node_id: str,
        group_id: str,
        edge_type: str | None = None,
        limit: int = 25,
    ) -> list[Edge]:
        """Fetch every ``RELATES_TO`` edge incident on ``node_id`` within ``group_id``.

        Cypher assumptions (from Graphiti's data model):

        * Entity nodes carry properties ``uuid`` (string) and ``group_id``
          (string). The query matches a node by ``uuid`` *and* ``group_id``
          to keep namespaces strictly isolated, even though uuids should
          be globally unique.
        * Relationships connecting entities are typed ``RELATES_TO`` and
          carry their own ``uuid``, ``group_id``, ``name`` (the edge type
          label, e.g. ``TREATS``), ``fact`` (a natural-language summary
          produced by Graphiti), and ``attributes`` (a property map of
          edge-specific structured fields).
        * ``startNode(r)`` and ``endNode(r)`` are used for direction-
          agnostic traversal — the query returns edges incident on the
          node regardless of orientation. Filtering by ``edge_type``
          matches against ``r.name``.

        These assumptions track Graphiti core's current schema; if
        Graphiti changes its property names or relationship type, this
        query is the single place that needs updating.
        """
        try:
            async with self._neo4j.driver.session(database=self._neo4j.database) as session:
                records = await session.run(
                    _FETCH_EDGES_CYPHER,
                    uuid=node_id,
                    group_id=group_id,
                    edge_type=edge_type,
                    limit=limit,
                )
                rows = [record.data() async for record in records]
        except Exception:
            logger.exception("fetch_edges failed for node=%s group=%s", node_id, group_id)
            return []

        edges: list[Edge] = []
        for row in rows:
            attrs: dict[str, Any] = {}
            raw_attrs = row.get("attributes")
            if isinstance(raw_attrs, dict):
                attrs.update(raw_attrs)
            fact = row.get("fact")
            if fact:
                attrs["fact"] = fact
            episodes = row.get("episodes")
            if isinstance(episodes, list):
                attrs["episodes"] = list(episodes)
            edges.append(
                Edge(
                    id=row["uuid"],
                    edge_type=row.get("edge_type") or "RELATES_TO",
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    properties=attrs,
                )
            )
        return edges

    async def search_edges(
        self,
        query: str,
        group_id: str,
        limit: int = 20,
        center_node_uuid: str | None = None,
    ) -> list[Edge]:
        """Hybrid search for fact edges, optionally reranked by graph distance.

        Wraps Graphiti's ``search()`` API:

        * ``center_node_uuid is None`` → ``EDGE_HYBRID_SEARCH_RRF`` (BM25 +
          cosine over fact text, fused via reciprocal-rank-fusion).
        * ``center_node_uuid is not None`` → ``EDGE_HYBRID_SEARCH_NODE_DISTANCE``
          (same hybrid retrieval, then reranked by hop distance from the
          provided node).

        Two-stage retrieval (call without center, then again centered on the
        top result's source node) is the canonical Graphiti pattern for
        getting both query-relevance and graph-locality.
        """
        try:
            results = await self._graphiti.graphiti.search(
                query=query,
                group_ids=[group_id],
                num_results=limit,
                center_node_uuid=center_node_uuid,
            )
        except Exception:
            logger.exception(
                "search_edges failed for query=%r group=%s center=%s",
                query,
                group_id,
                center_node_uuid,
            )
            return []

        edges: list[Edge] = []
        for entity_edge in results:
            attrs: dict[str, Any] = {}
            raw_attrs = getattr(entity_edge, "attributes", None)
            if isinstance(raw_attrs, dict):
                attrs.update(raw_attrs)
            if entity_edge.fact:
                attrs["fact"] = entity_edge.fact
            episodes = getattr(entity_edge, "episodes", None)
            if isinstance(episodes, list):
                attrs["episodes"] = list(episodes)
            # Temporal data: surfaced to the generator so it can reason about
            # *when* a fact applies, not just *whether* it exists.
            for tkey in ("valid_at", "invalid_at", "reference_time", "created_at"):
                tval = getattr(entity_edge, tkey, None)
                if tval is not None:
                    attrs[tkey] = tval.isoformat() if hasattr(tval, "isoformat") else str(tval)
            edges.append(
                Edge(
                    id=entity_edge.uuid,
                    edge_type=entity_edge.name or "RELATES_TO",
                    source_id=entity_edge.source_node_uuid,
                    target_id=entity_edge.target_node_uuid,
                    properties=attrs,
                )
            )
        return edges

    async def fetch_node_names(
        self,
        uuids: list[str],
        group_id: str,
    ) -> dict[str, str]:
        """Resolve a batch of entity UUIDs to their ``name`` properties."""
        if not uuids:
            return {}
        try:
            async with self._neo4j.driver.session(database=self._neo4j.database) as session:
                records = await session.run(
                    _FETCH_NODE_NAMES_CYPHER,
                    uuids=list(set(uuids)),
                    group_id=group_id,
                )
                rows = [record.data() async for record in records]
        except Exception:
            logger.exception(
                "fetch_node_names failed for %d uuids in group=%s",
                len(uuids),
                group_id,
            )
            return {}
        return {row["uuid"]: row.get("name") or "" for row in rows if row.get("uuid")}

    async def fetch_episode_chunks(
        self,
        episode_uuids: list[str],
        group_id: str,
    ) -> dict[str, dict[str, str]]:
        """Fetch source episode contents by UUID, scoped to ``group_id``.

        Returns ``{uuid: {"name": ..., "content": ...}}``. Episodes outside
        the group or missing are omitted. Empty input → empty dict.
        """
        if not episode_uuids:
            return {}
        try:
            async with self._neo4j.driver.session(database=self._neo4j.database) as session:
                records = await session.run(
                    _FETCH_EPISODE_CHUNKS_CYPHER,
                    uuids=list(set(episode_uuids)),
                    group_id=group_id,
                )
                rows = [record.data() async for record in records]
        except Exception:
            logger.exception(
                "fetch_episode_chunks failed for %d uuids in group=%s",
                len(episode_uuids),
                group_id,
            )
            return {}
        out: dict[str, dict[str, str]] = {}
        for row in rows:
            uuid = row.get("uuid")
            if not uuid:
                continue
            out[uuid] = {
                "name": row.get("name") or "",
                "content": row.get("content") or "",
            }
        return out
