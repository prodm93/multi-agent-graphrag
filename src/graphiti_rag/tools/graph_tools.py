"""Composition of the focused graph query tools.

Bundles :class:`GraphEdgeTools`, :class:`GraphNodeTools` and
:class:`GraphEpisodeTools` behind a single object so agents can hold one
dependency. Each sub-tool is reachable as ``.edges``, ``.nodes`` and
``.episodes`` respectively. Future agents that need only a subset should
take those sub-tools directly rather than this aggregate.
"""
from __future__ import annotations

from graphiti_rag.graph.graphiti_client import GraphitiClient
from graphiti_rag.graph.neo4j_client import Neo4jClient
from graphiti_rag.tools.graph.edges import GraphEdgeTools
from graphiti_rag.tools.graph.episodes import GraphEpisodeTools
from graphiti_rag.tools.graph.nodes import GraphNodeTools


class GraphTools:
    """Aggregate of edge, node and episode query tools."""

    def __init__(self, graphiti: GraphitiClient, neo4j: Neo4jClient) -> None:
        self.edges = GraphEdgeTools(graphiti, neo4j)
        self.nodes = GraphNodeTools(graphiti, neo4j)
        self.episodes = GraphEpisodeTools(neo4j)
