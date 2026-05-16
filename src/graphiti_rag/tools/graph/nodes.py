"""Node queries over Graphiti + Neo4j.

Pure I/O. ``search_nodes`` delegates to Graphiti's hybrid node search
(``NODE_HYBRID_SEARCH_RRF`` — BM25 + cosine + BFS, fused and reranked).
``find_entity_by_name`` does a case-insensitive substring match against
``Entity.name`` and is used by the planner's ``entity_lookup`` strategy.
``fetch_node_names`` resolves UUID batches back to display names.
"""
from __future__ import annotations

import logging

from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

from graphiti_rag.domain import Node
from graphiti_rag.graph.graphiti_client import GraphitiClient
from graphiti_rag.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


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


class GraphNodeTools:
    """Node-side reads against the knowledge graph."""

    def __init__(self, graphiti: GraphitiClient, neo4j: Neo4jClient) -> None:
        self._graphiti = graphiti
        self._neo4j = neo4j

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

    async def find_entity_by_name(
        self,
        entity_name: str,
        group_id: str,
        limit: int = 5,
    ) -> list[Node]:
        """Find entity nodes by case-insensitive substring match on ``name``.

        Shorter matches sort first — a literal hit beats a long string
        that merely contains the query.
        """
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
