# Multi-Agent GraphRAG

Vanilla RAG--or even RAG with bells and whistles such as hybrid search and reranking--can be great for extracting straightforward single-fact chunks from a single document in your text data. But what about when your data is in a niche and complex domain, you have tons of documents, mitigating hallucination and provenance tracing are critical priorities, and you want to be able to hop across the chunks across these documents to surface truly informative, abstractive and nuanced relational information?

This is where GraphRAG--an advanced framework which leverages knowledge graphs to enhance LLMs' understanding of complex and private datasets--shines. However, constructing static knowledge graphs from scratch every time you need to add new documents is computationally (and prohibitely) expensive. Enter agentic GraphRAG: a framework that allows for agents to decide how to traverse graphs best based on the query, harness temporal graphs with real-time data and use time as a first-class data point. However, off-the-shelf agentic graphRAG frameworks are still prone to creating generic relationship edges such as 'mentions' and 'relates to', which are unlikely to be very informative for complex domains.

---

To bridge these gaps, I created Multi-Agent GraphRAG--a domain-agnostic agentic GraphRAG pipeline chained up to multiple useful LLM-based agents:

- Given a corpus of complex or niche documents, `OntologyAgent` infers and creates a serialisable `OntologyDefinition` custom-tailored to your data; namely, entity TYPE definitions, edge TYPE definitions, and allowed edge mappings (type/category definitions only, not actual graph nodes or relationships). 
- `GraphAgent` then drives Graphiti to build a Neo4j knowledge graph from the documents under that ontology, and answers natural-language queries grounded in the graph. 
- During query answering, `RetrievalPlannerAgent` selects the most appropriate graph-retrieval strategy for the user query instead of forcing a single fixed traversal. 
- `ContextAgent` then executes that plan, using edge-first hybrid fact search, centred graph-distance reranking, or entity-focused lookup to build compact graph-derived context for `GeneratorAgent`. 

If planning fails or returns no usable context, the system automatically falls back to the deterministic edge-first hybrid search with centred reranking. Retrieved context includes surfaced relationship facts, temporal metadata where available, and source episode excerpts for grounding.

Users supply their own Neo4j AuraDB credentials and OpenAI API key through the app sidebar. Credentials live in session state only and are never written to disk or stored server-side.

### Work-in-Progress

- Improve the playbook source content for retrieval planning further
- Make the retrieval planning agent system prompt even more robust & add pointed instructions
- Include support for non-openAI models
- Host the app so end users no longer need to clone-and-run (see _Observability & privacy_ below)

## Observability & privacy

The app supports LangSmith for agent tracing and observability. In this clone-and-run distribution, however, tracing is **developer-side only**. The FastAPI server only emits traces if its own process environment contains a valid `LANGSMITH_API_KEY`. End users running the app on their own machine do not have such a key and so no data ever leaves their machine.

A privacy modal is included that offers three data-sharing tiers (full, anonymised, metadata-only) backed by LangSmith's `anonymizer` callable and `hide_inputs` / `hide_outputs` switches. The choice flows end-to-end through `POST /api/consent` → `AppState.consent_tier` → the per-agent `wrap_openai(..., tracing_extra={"client": ls_client})` calls. In a clone-and-run build the wiring is _shadow live_ — fully exercised, but inert because no traces are being emitted in the first place. It becomes load-bearing the moment the app is hosted with a real LangSmith key (a future direction; see Work-in-Progress).

The modal is soft; it appears once on first visit, can be dismissed without choosing, and can be reopened any time from the **Privacy** link in the sidebar.


## Run the app

Creating a Neo4j account and setting up an AuraDB instance on the console is the easiest way to get set up with the knowledge graph. At the time of creation, you will be given the option to download your credentials to disk in a .txt file. It's strongly suggested to do so since you're only shown your credentials once, and it will come in handy for connectting to the database on the app. **Please ensure your Neo4j database/AuraDB instance is unpaused prior to querying.**

1. Install Docker Desktop (or OrbStack, this repo author's personal lightweight favourite)
2. `cd` into the repo root in terminal, then `docker compose up --build`
3. Open http://localhost:5173
4. In the UI:
   - Upload the Neo4j credentials `.txt` file OR enter credentials manually
   - Paste OpenAI API key (only OpenAI support provided for now; more coming soon)
   - Connect
   - Upload documents
   - Ask questions

A sample credentials file pointing at the bundled local Neo4j is available at `local-neo4j-creds.example.txt`. For Neo4j AuraDB, upload the `.txt` file Aura provides on database creation.

## Sanity Checks/Some Tips

Neo4j's native graph visualiser (Bloom) does not make the incorporation of custom edge relationship names obvious. While the graph may still show generic 'MENTIONS' and 'RELATES_TO' labels on edges, you can set the caption type to the name of the edge or double-check that your custom edges were inferred and incorporated by running the following Cypher query:
```
  MATCH ()-[r:RELATES_TO]->()
  RETURN DISTINCT r.name AS edge_type, count(*) AS n
  ORDER BY n DESC
  ```

## Naming

| | |
|---|---|
| Project / display name | `multi-agent-graphrag` |
| Distribution name (`pyproject.toml`) | `multi-agent-graphrag` |
| Python import package | `graphiti_rag` |

So Python code uses `from graphiti_rag.config import Config`, but the installable distribution is `multi-agent-graphrag`.

## Stack

- **Orchestration**: LangGraph
- **Knowledge graph**: Graphiti + Neo4j (AuraDB in production, Docker Compose locally)
- **Schemas**: Pydantic v2
- **Backend**: FastAPI (`uvicorn graphiti_rag.api.app:app`)
- **Frontend**: React + Vite + TypeScript
- **Python**: 3.11+

## Developer mode

For local development with hot reload, run only Neo4j in Docker and start the backend and frontend on the host:

```bash
# Start a local Neo4j
docker compose up -d neo4j

# Install the Python package (src-layout)
pip install -e .

# Configure environment
cp .env.example .env

# Run the backend
uvicorn graphiti_rag.api.app:app --reload

# Run the frontend
cd frontend && npm install && npm run dev
```

See `CLAUDE.md` for architecture and `SKILL.md` for implementation patterns.
