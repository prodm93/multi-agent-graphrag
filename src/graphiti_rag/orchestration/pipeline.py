"""LangGraph pipeline wiring all agents.

Ingest and query are separate user actions and therefore separate graphs:

* ``ingest(doc_paths)`` — ontology_inference → graph_build
* ``query(question)``  — context → generate

The :class:`Pipeline` owns the per-request resource lifecycle: each public
entry-point (``ingest``/``query``) opens the Neo4j and Graphiti clients,
runs the relevant LangGraph, and closes the clients in a ``finally`` block.

Index/constraint creation is performed only on the ingest path (after
``initialise``). The query path skips it because read-only traffic must
not pay the per-request overhead of a schema-management call.
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from graphiti_rag.agents.context_agent import ContextAgent
from graphiti_rag.agents.generator_agent import GeneratorAgent
from graphiti_rag.agents.graph_agent import GraphAgent
from graphiti_rag.agents.ontology_agent import OntologyAgent
from graphiti_rag.agents.retrieval_planner_agent import RetrievalPlannerAgent
from graphiti_rag.config import Config
from graphiti_rag.domain import IngestResult
from graphiti_rag.graph.graphiti_client import GraphitiClient
from graphiti_rag.graph.neo4j_client import Neo4jClient
from graphiti_rag.playbook import PLAYBOOK
from graphiti_rag.schemas.ontology import OntologyDefinition
from graphiti_rag.tools.document_loader import DocumentLoader
from graphiti_rag.tools.graph_tools import GraphTools

logger = logging.getLogger(__name__)


class IngestState(TypedDict):
    doc_paths: list[str]
    ontology: OntologyDefinition
    ingest_result: IngestResult


class QueryState(TypedDict):
    query: str
    context: str
    answer: str


class Pipeline:
    """Wires agents into two LangGraph StateGraphs (ingest, query)."""

    def __init__(
        self,
        config: Config,
        ontology_agent: OntologyAgent,
        graph_agent: GraphAgent,
        context_agent: ContextAgent,
        generator_agent: GeneratorAgent,
        neo4j_client: Neo4jClient,
        graphiti_client: GraphitiClient,
    ) -> None:
        self._config = config
        self._ontology_agent = ontology_agent
        self._graph_agent = graph_agent
        self._context_agent = context_agent
        self._generator_agent = generator_agent
        self._neo4j = neo4j_client
        self._graphiti = graphiti_client
        self._ingest_graph = self._build_ingest_graph()
        self._query_graph = self._build_query_graph()

    @classmethod
    def from_config(
        cls,
        config: Config,
        langsmith_client: object | None = None,
    ) -> "Pipeline":
        """Convenience factory — wires dependencies from a Config object.

        ``langsmith_client``, when supplied, is passed to the LLM-calling
        agents (planner and generator) so each ``wrap_openai`` call attaches
        the per-tier privacy filters via ``tracing_extra``.
        """
        neo4j = Neo4jClient(config)
        graphiti = GraphitiClient(config, neo4j)
        loader = DocumentLoader()
        graph_tools = GraphTools(graphiti, neo4j)
        try:
            planner_agent = RetrievalPlannerAgent(
                config, PLAYBOOK, langsmith_client=langsmith_client
            )
        except Exception:
            logger.exception(
                "Retrieval planner construction failed; using deterministic retrieval"
            )
            planner_agent = None
        return cls(
            config=config,
            ontology_agent=OntologyAgent(config, loader),
            graph_agent=GraphAgent(config, graphiti, loader),
            context_agent=ContextAgent(config, graph_tools, planner_agent),
            generator_agent=GeneratorAgent(config, langsmith_client=langsmith_client),
            neo4j_client=neo4j,
            graphiti_client=graphiti,
        )

    def _build_ingest_graph(self) -> StateGraph:
        builder = StateGraph(IngestState)
        builder.add_node("ontology_inference", self._run_ontology)
        builder.add_node("graph_build", self._run_graph_build)
        builder.set_entry_point("ontology_inference")
        builder.add_edge("ontology_inference", "graph_build")
        builder.add_edge("graph_build", END)
        return builder.compile()

    def _build_query_graph(self) -> StateGraph:
        builder = StateGraph(QueryState)
        builder.add_node("context", self._run_context)
        builder.add_node("generate", self._run_generate)
        builder.set_entry_point("context")
        builder.add_edge("context", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    async def ingest(self, doc_paths: list[str]) -> IngestResult:
        """Ingest documents into the knowledge graph and return aggregate result.

        Opens the Neo4j and Graphiti clients on entry, ensures indices
        exist (write path only), runs the ingest graph, and closes both
        clients in ``finally``.
        """
        logger.info("Pipeline.ingest: %d documents", len(doc_paths))
        try:
            await self._neo4j.connect()
            await self._graphiti.initialise()
            await self._graphiti.ensure_indices_and_constraints()
            final_state = await self._ingest_graph.ainvoke(
                {
                    "doc_paths": doc_paths,
                    "ontology": OntologyDefinition(),
                    "ingest_result": IngestResult(successes=0, failures=0),
                }
            )
            return final_state["ingest_result"]
        finally:
            await self._graphiti.close()
            await self._neo4j.close()

    async def query(self, query: str) -> str:
        """Answer a natural-language query grounded in the graph.

        Read-only path: opens the clients, runs the query graph, closes
        both in ``finally``. Does **not** rebuild indices on every call.
        """
        logger.info("Pipeline.query: %d-char query", len(query))
        try:
            await self._neo4j.connect()
            await self._graphiti.initialise()
            result = await self._query_graph.ainvoke(
                {"query": query, "context": "", "answer": ""}
            )
            return result["answer"]
        finally:
            await self._graphiti.close()
            await self._neo4j.close()

    async def _run_ontology(self, state: IngestState) -> IngestState:
        ontology = await self._ontology_agent.run(state["doc_paths"])
        return {**state, "ontology": ontology}

    async def _run_graph_build(self, state: IngestState) -> IngestState:
        result = await self._graph_agent.run(state["doc_paths"], state["ontology"])
        return {**state, "ingest_result": result}

    async def _run_context(self, state: QueryState) -> QueryState:
        context = await self._context_agent.run(state["query"])
        return {**state, "context": context}

    async def _run_generate(self, state: QueryState) -> QueryState:
        answer = await self._generator_agent.run(state["query"], state["context"])
        return {**state, "answer": answer}
