"""Retrieval strategy playbook used by the planner agent.

A single text block describing the available retrieval strategies, when
each is the right choice, and the Graphiti API surface they wrap. The
:class:`RetrievalPlannerAgent` feeds this into its LLM prompt at every
query so the agent can pick a strategy without us hardcoding a router.

This module is intentionally minimal. A future revision may regenerate
the playbook dynamically from web search; until that path is built and
tested, no toggle exists.
"""
from __future__ import annotations


PLAYBOOK = """\
# Retrieval Strategies for Graphiti-backed GraphRAG

You pick ONE strategy per user query. Each strategy wraps a Graphiti or
Neo4j operation. If none fit cleanly, default to `edge_hybrid` — it is the
safest general-purpose option.

## Strategies

### edge_hybrid
Wraps `graphiti.search(query)` with `EDGE_HYBRID_SEARCH_RRF` (BM25 +
cosine over fact text, fused via reciprocal-rank fusion). Returns the
top-K fact edges ranked by relevance to the query string.

Use when:
- Open-ended question without a single named focal entity
  (e.g. "what side effects do checkpoint inhibitors have?")
- The user wants the most query-relevant facts regardless of which entity
  they hang off
- You're unsure which strategy fits

### centered_rerank
Wraps `graphiti.search(query, center_node_uuid=...)` with
`EDGE_HYBRID_SEARCH_NODE_DISTANCE`. Two LLM-side calls: first an
`edge_hybrid` to find an anchor, then a rerank around the anchor's source
entity by graph distance.

Use when:
- The query implies a focal entity and asks for *related* / *associated*
  / *connected* / *similar* information
  (e.g. "what other drugs work like pembrolizumab?")
- The user already named an entity but wants its neighbourhood, not just
  literal mentions

### entity_lookup
Looks up an entity by name in the graph (case-insensitive partial match),
then returns the edges incident on it.

Use when:
- The query names a specific entity and asks for facts ABOUT that entity
  (e.g. "what is keytruda approved for?")
- The user wants completeness over a single subject, not relevance ranking

Required param: `entity_name` (the name as the user wrote it).

## Output format

Return ONLY a JSON object with this shape:

{
  "strategy": "edge_hybrid" | "centered_rerank" | "entity_lookup",
  "params": { ... strategy-specific params ... },
  "reason": "one short sentence explaining the pick"
}

Examples:

Query: "what's the most common side effect of pembrolizumab?"
{
  "strategy": "edge_hybrid",
  "params": {},
  "reason": "Open question over fact text — pure relevance wins."
}

Query: "what other drugs work like ipilimumab?"
{
  "strategy": "centered_rerank",
  "params": {},
  "reason": "Focal entity (ipilimumab) plus a neighbourhood-style ask."
}

Query: "tell me everything about KEYNOTE-006"
{
  "strategy": "entity_lookup",
  "params": {"entity_name": "KEYNOTE-006"},
  "reason": "Single named subject, exhaustive ask."
}
"""
