# MCP Tool-Retrieval Gateway

**A self-hostable MCP proxy that exposes only the top-k semantically relevant tools per query — instead of every tool from every server.**

[![tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)](#running-the-tests) [![python](https://img.shields.io/badge/python-3.11%2B-blue)](#requirements) [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The problem

The [Model Context Protocol](https://modelcontextprotocol.io) (MCP) lets an LLM-based client connect to many tool servers — filesystem, GitHub, a database, a browser, your internal APIs. But there is a cost that grows with every server you add:

> **An MCP client loads the *full* tool definitions from *every* connected server into the context window on *every* request** — names, descriptions, and complete JSON input schemas — before the user has typed a single word.

A handful of servers can easily contribute **50–100+ tool definitions**. Rich JSON schemas are verbose, and in practice this commonly burns **~20–40% of the context window as fixed overhead** on every turn. That overhead is:

- **Paid on every request**, whether or not any of those tools are relevant.
- **Mostly wasted** — a typical query needs 1–3 tools, not 80.
- **Money and latency** — more prompt tokens on every call, and a larger prompt the model must attend to.

For a single query like *"convert 100 USD to EUR"*, the model does not need the weather tools, the calendar tools, or the git tools. It needs one.

## The solution

This gateway sits **between the client and the upstream MCP servers** as a proxy. It:

1. **Embeds** every upstream tool definition once, at startup, into a vector store.
2. On each `tools/list`, takes the **query/context** and returns **only the top-k** tools whose embeddings are most similar — not the whole catalogue.
3. On `tools/call`, **routes** the invocation back to the upstream server that actually owns that tool.

The client sees a small, query-relevant tool list. The context overhead drops from *"all tools, always"* to *"k tools, on demand."* With the bundled example (16 tools) and `k=3`, that is an **81% reduction in tool-definition tokens** for a given query — and the ratio only improves as you connect more servers.

```
$ python examples/demo.py

Total tools       : 16  (what a naive client loads every request)
Exposed per query : 3   (top-k relevant tools)
Tool-definition reduction: 16 -> 3  (~81% fewer)

query: "what's the weather forecast for tomorrow"
   1. get_forecast           [weather ] score=0.375
   2. get_stock_price        [finance ] score=0.364
   3. list_events            [calendar] score=0.134
```

## Architecture

```
                          MCP Tool-Retrieval Gateway
                 ┌────────────────────────────────────────────┐
                 │                                            │
   MCP client    │   FastAPI  (JSON-RPC 2.0)                  │      Upstream MCP servers
  ┌──────────┐   │   ┌────────────────────────┐               │      ┌───────────────────┐
  │          │   │   │ POST /                 │               │      │ weather   (tools) │
  │  tools/  │──────▶│  method: tools/list    │               │  ┌──▶│ finance   (tools) │
  │  list    │   │   │  method: tools/call    │               │  │   │ files     (tools) │
  │  (query) │◀──────│                        │               │  │   │ email     (tools) │
  │          │   │   └───────────┬────────────┘               │  │   │ calendar  (tools) │
  │  tools/  │   │               │                            │  │   │ devtools  (tools) │
  │  call    │──────┐            ▼                            │  │   └───────────────────┘
  └──────────┘   │  │   ┌──────────────────┐                  │  │
                 │  │   │  ToolRegistry     │  routes call ────┼──┘
                 │  │   │  ┌─────────────┐  │                  │
                 │  │   │  │  Retriever   │  │                  │
                 │  │   │  │  embedder ──▶│  │  top-k tools     │
                 │  └──▶│  │  vectorstore │  │                  │
                 │      │  └─────────────┘  │                  │
                 │      └──────────────────┘                  │
                 │        loads at startup from config.yaml    │
                 └────────────────────────────────────────────┘

  index (startup):  ToolDef ─▶ embedding_text ─▶ embed() ─▶ VectorStore.add()
  query (per call): text     ─▶ embed()         ─▶ VectorStore.search(k) ─▶ ToolDefs
```

**Module map**

| Module | Responsibility |
|---|---|
| `mcp_router/config.py` | Parse the YAML/JSON config: which upstreams, which tools. |
| `mcp_router/models.py` | `ToolDef` — an MCP tool + the upstream that owns it. |
| `mcp_router/embedder.py` | Pluggable embedders: offline `HashingEmbedder` (default) or a real one. |
| `mcp_router/vectorstore.py` | In-memory NumPy cosine-similarity store. |
| `mcp_router/retrieval.py` | `Retriever` — indexes tools, returns top-k for a query. |
| `mcp_router/registry.py` | Ties it together; owns retrieval + upstream routing. |
| `mcp_router/upstream.py` | Where `tools/call` executes (mock upstream for the MVP). |
| `mcp_router/server.py` | FastAPI app exposing MCP-shaped JSON-RPC. |

## Requirements

- Python 3.11+
- `fastapi`, `uvicorn`, `numpy`, `pyyaml` (see `requirements.txt`)

The default embedder is fully offline and needs no model download or API key.

## Quick start (offline)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. See the reduction, end to end
python examples/demo.py

# 3. Run the gateway as an HTTP server
export MCP_ROUTER_CONFIG=config.example.yaml
uvicorn mcp_router.server:app_from_env --factory --port 8000
```

Then call it with plain JSON-RPC:

```bash
# tools/list with a query -> only the top-k relevant tools
curl -s localhost:8000/ -H 'content-type: application/json' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "tools/list",
  "params": {"query": "convert dollars to euros", "k": 3}
}'

# tools/call -> routed to the upstream that owns the tool
curl -s localhost:8000/ -H 'content-type: application/json' -d '{
  "jsonrpc": "2.0", "id": 2, "method": "tools/call",
  "params": {"name": "convert_currency", "arguments": {"amount": 100}}
}'
```

Omit `query` from `tools/list` and the gateway returns *all* tools — behaving as a transparent proxy.

### With Docker

```bash
docker build -t mcp-tool-router .
docker run -p 8000:8000 mcp-tool-router
# or point it at your own config:
docker run -p 8000:8000 -e MCP_ROUTER_CONFIG=/app/my.yaml -v $PWD/my.yaml:/app/my.yaml mcp-tool-router
```

## Configuration

Copy `config.example.yaml` and describe your upstreams. Each server has a `name`, a `transport`, and (for the offline `mock` transport) inline tool definitions:

```yaml
servers:
  - name: finance
    transport: mock
    tools:
      - name: convert_currency
        description: Convert a monetary amount from one currency to another.
        inputSchema:
          type: object
          properties:
            amount: { type: number }
```

## Plugging in a real embedder

The default `HashingEmbedder` matches on **lexical overlap** — it is deterministic and offline, ideal for tests and demos, but it does not understand synonyms (e.g. *"schedule a meeting"* will not strongly match a tool described as *"create a calendar event"*). For true semantic matching, switch to a real model — no code change, just environment variables:

```bash
pip install sentence-transformers
export MCP_ROUTER_EMBEDDER=sentence-transformers
export MCP_ROUTER_ST_MODEL=all-MiniLM-L6-v2   # optional; this is the default
```

To wire in a hosted embedding API instead, implement the `Embedder` interface (one method, `embed(texts) -> np.ndarray`) and return it from `get_embedder()`. Everything downstream depends only on that interface.

## Running the tests

```bash
pip install -r requirements.txt
pytest
```

The suite is **offline and deterministic** (29 tests) and covers: the embedder, the vector store, retrieval correctness, that `tools/list` returns only top-k, and that `tools/call` routes to the right upstream.

## MVP scope vs. roadmap

This is an honest v0.1 — a small, working, tested core. Here is exactly what is real today and what is deliberately deferred.

**Real in v0.1 (implemented + tested):**

- Config-driven tool registry (YAML/JSON), multiple upstreams.
- Pluggable embedder (offline hashing default; real embedder via env var).
- NumPy cosine vector store with exact top-k search.
- MCP-shaped JSON-RPC `tools/list` (query → top-k) and `tools/call` (routing).
- FastAPI HTTP server + `/healthz`, Docker image, runnable demo.

**Deferred (documented next milestones):**

- **Real upstream MCP transport.** `tools/call` currently routes to a `MockUpstream` that echoes the call. The routing logic — *which* upstream owns a tool — is real and tested; what is mocked is the wire transport to that server. The next milestone is a `StdioUpstream` that launches a real MCP server as a subprocess and speaks JSON-RPC over stdio via the official [`mcp`](https://pypi.org/project/mcp/) Python SDK, plus live tool discovery via the upstream's own `tools/list`. This is intentionally **not** faked here.
- **Approximate vector index** (FAISS/hnswlib) for very large tool catalogues — the current exact search is the right choice for tens–hundreds of tools.
- **Reranking / hybrid retrieval** (combine lexical + semantic scores).
- **Full MCP server compliance** (`initialize` handshake, notifications, resources/prompts) so standard MCP clients can connect directly over stdio/SSE.
- **Caching** of query→tool-set results.

## License

MIT — see [LICENSE](LICENSE).
