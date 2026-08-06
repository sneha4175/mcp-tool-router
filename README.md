# MCP Tool-Retrieval Gateway

**A self-hostable MCP proxy that exposes only the top-k semantically relevant tools per query — instead of every tool from every server.**

[![tests](https://img.shields.io/badge/tests-32%20passing-brightgreen)](#running-the-tests) [![python](https://img.shields.io/badge/python-3.11%2B-blue)](#requirements) [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **v0.2** — real MCP transport over the official [`mcp`](https://pypi.org/project/mcp/) SDK (the gateway is now a real MCP **client** *and* a real MCP **server**), and a real local semantic embedder by default. See [what's verified vs roadmap](#whats-verified-vs-roadmap).

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

1. Connects to each upstream MCP server and **discovers its real tools**.
2. **Embeds** every tool definition once, at startup, into a vector store.
3. Per query, returns **only the top-k** tools whose embeddings are most similar — not the whole catalogue.
4. On a call, **routes** the invocation back to the upstream server that actually owns that tool and returns its real result.

The client sees a small, query-relevant tool list. The context overhead drops from *"all tools, always"* to *"k tools, on demand."*

### Real end-to-end run (v0.2)

Connected to **two real MCP servers over stdio** — the bundled example server plus the official `@modelcontextprotocol/server-filesystem` via `npx` — using the default `sentence-transformers` embedder:

```
$ python examples/real_mcp_demo.py

Upstream MCP servers : 2  (real, live over stdio)
Real tools discovered: 18  (a naive client would load all of these)
Exposed per query    : 3  (top-k relevant)
Reduction            : 18 -> 3  (~83% fewer)

query: 'read the contents of a text file'
   1. read_text_file         [filesystem] score=0.711
   2. read_file              [filesystem] score=0.674
   3. read_multiple_files    [filesystem] score=0.496

query: 'add two numbers together'
   1. add_numbers            [example   ] score=0.784
   2. move_file              [filesystem] score=0.111
   3. reverse_text           [example   ] score=0.092

----------------------------------------------------------------------
tools/call reverse_text('gateway') -> [example] yawetag
```

Those tools were **fetched live from the real servers**, and the final line is a **real call proxied to the upstream** that owns `reverse_text` — its actual output, `yawetag`, returned back.

## Architecture

```
                          MCP Tool-Retrieval Gateway
                 ┌────────────────────────────────────────────┐
   MCP host      │                                            │   Upstream MCP servers
 (Claude Desktop)│   MCP server (stdio, mcp SDK)              │   (real, over stdio)
  ┌──────────┐   │   ┌────────────────────────┐               │   ┌────────────────────┐
  │find_tools│──────▶│  find_tools(query,k)    │  top-k        │   │ example  (SDK)     │
  │          │◀──────│                        │◀──────┐        │┌─▶│ filesystem (npx)   │
  │call_tool │──────▶│  call_tool(name,args)   │       │        ││  │ ...your servers... │
  └──────────┘   │   └───────────┬────────────┘       │        ││  └────────────────────┘
                 │               │            ┌────────┴─────┐  ││
   HTTP client   │   FastAPI     ▼            │ ToolRegistry │  ││
  ┌──────────┐   │   ┌────────────────────┐   │  Retriever   │  ││
  │tools/list│──────▶│ POST / (JSON-RPC)  │──▶│  embedder    │  ││
  │  (query) │◀──────│ tools/list+query   │   │  vectorstore │  ││
  │tools/call│──────▶│ tools/call         │───┼─ routes call ─┼──┘│
  └──────────┘   │   └────────────────────┘   │  Upstream ───┼───┘
                 │                            └──────────────┘   (MockUpstream | StdioUpstream)
                 └────────────────────────────────────────────┘

  index (startup):  connect upstream ─▶ discover tools ─▶ embed() ─▶ VectorStore.add()
  query:            text ─▶ embed() ─▶ VectorStore.search(k) ─▶ ToolDefs
  call:             name ─▶ owning Upstream.call() ─▶ real result
```

Two front doors, one core:

- **MCP server (stdio)** — what a standard MCP host (Claude Desktop) connects to. See [Run as an MCP server](#run-as-an-mcp-server-claude-desktop).
- **HTTP JSON-RPC** — a convenient HTTP surface whose `tools/list` takes an extra `query` param. See [Run as an HTTP server](#run-as-an-http-server).

**Module map**

| Module | Responsibility |
|---|---|
| `mcp_router/config.py` | Parse the YAML/JSON config: which upstreams (`mock` or `stdio`), which tools. |
| `mcp_router/models.py` | `ToolDef` — an MCP tool + the upstream that owns it. |
| `mcp_router/embedder.py` | Pluggable embedders: real `sentence-transformers` (default) or offline `HashingEmbedder`. |
| `mcp_router/vectorstore.py` | In-memory NumPy cosine-similarity store. |
| `mcp_router/retrieval.py` | `Retriever` — indexes tools, returns top-k for a query. |
| `mcp_router/upstream.py` | `MockUpstream` (offline) and `StdioUpstream` (**real** MCP client over stdio). |
| `mcp_router/registry.py` | Ties it together; discovers tools from upstreams, owns retrieval + routing. |
| `mcp_router/mcp_server.py` | The gateway as a **real MCP server** (`find_tools` + `call_tool`). |
| `mcp_router/server.py` | FastAPI app exposing MCP-shaped JSON-RPC over HTTP. |

## Requirements

- Python 3.11+
- `fastapi`, `uvicorn`, `numpy`, `pyyaml`, `mcp` (see `requirements.txt`)
- Optional: `sentence-transformers` for the default semantic embedder; Node/`npx` only if you point at npx-launched upstream servers.

## Quick start

```bash
# 1. Install (lean runtime + the real MCP SDK)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2a. Semantic default: install the embedder (downloads ~80MB on first use)
pip install sentence-transformers

# 2b. ...or run fully offline/deterministic instead:
export MCP_ROUTER_EMBEDDER=hashing

# 3. See the reduction against REAL MCP servers, end to end
python examples/real_mcp_demo.py
```

`examples/demo.py` is the original offline demo over inline mock tools; `examples/real_mcp_demo.py` (above) connects to real MCP servers over stdio.

## Run as an MCP server (Claude Desktop)

The gateway serves the real MCP protocol over stdio. Because a vanilla `tools/list` has nowhere to put a query, it exposes the reduction through **progressive disclosure** — just two meta-tools, so a host loads *2* tool definitions instead of *N*:

- **`find_tools(query, k)`** — semantic search across every upstream; returns the top-k matching tool definitions.
- **`call_tool(name, arguments)`** — proxies a call to whichever upstream owns the tool and returns its real result.

Run it standalone:

```bash
MCP_ROUTER_CONFIG=config.stdio.example.yaml python -m mcp_router.mcp_server
```

Add it to Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tool-router": {
      "command": "python",
      "args": ["-m", "mcp_router.mcp_server"],
      "env": {
        "MCP_ROUTER_CONFIG": "/absolute/path/to/config.stdio.example.yaml",
        "MCP_ROUTER_EMBEDDER": "sentence-transformers"
      }
    }
  }
}
```

The host then loads two tools; the model calls `find_tools("…")` to discover what it needs, then `call_tool(...)` to run it — keeping context overhead constant no matter how many upstream servers you connect.

## Run as an HTTP server

```bash
export MCP_ROUTER_CONFIG=config.example.yaml
uvicorn mcp_router.server:app_from_env --factory --port 8000
```

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

The image is lean and defaults to the offline hashing embedder (instant start, no download):

```bash
docker build -t mcp-tool-router .
docker run -p 8000:8000 mcp-tool-router
# point it at your own config:
docker run -p 8000:8000 -e MCP_ROUTER_CONFIG=/app/my.yaml -v $PWD/my.yaml:/app/my.yaml mcp-tool-router
```

## Configuration

Each server has a `name` and a `transport`.

**`stdio`** — a real MCP server launched as a subprocess (`config.stdio.example.yaml`):

```yaml
servers:
  - name: filesystem
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    env: {}
  - name: example
    transport: stdio
    command: python
    args: ["examples/example_upstream_server.py"]
```

**`mock`** — offline; tools listed inline, calls echoed (`config.example.yaml`):

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

## Embedders

The embedder is selected by `MCP_ROUTER_EMBEDDER`:

| Value | What it is | When |
|---|---|---|
| `sentence-transformers` (**default**) | Real local model `all-MiniLM-L6-v2` (384-dim). Understands meaning, not just shared words — *"schedule a meeting"* matches *"create a calendar event"*. | Production / real semantic retrieval. Downloads ~80MB once, then cached. |
| `hashing` | Dependency-free, deterministic hashing-trick embedder. Lexical overlap only. | Tests, CI, air-gapped runs. No download. |

```bash
export MCP_ROUTER_EMBEDDER=sentence-transformers   # default
export MCP_ROUTER_ST_MODEL=all-MiniLM-L6-v2         # optional; this is the default
# or, fully offline:
export MCP_ROUTER_EMBEDDER=hashing
```

To wire in a hosted embedding API, implement the `Embedder` interface (one method, `embed(texts) -> np.ndarray`) and return it from `get_embedder()`. Everything downstream depends only on that interface.

## Running the tests

```bash
pip install -r requirements.txt
pytest
```

The suite is **offline and deterministic** by design: it forces `MCP_ROUTER_EMBEDDER=hashing` and never downloads a model. It includes real MCP-transport integration tests that launch the bundled example server as a subprocess and speak the actual protocol to it (both the gateway-as-client and gateway-as-server paths).

- **32 passing, 1 skipped** locally. The skipped test connects to the official `@modelcontextprotocol/server-filesystem` via `npx` (needs Node + network); enable it with `MCP_ROUTER_RUN_NPX_TESTS=1 pytest`.

## What's verified vs roadmap

Honest status for v0.2.

**Verified end-to-end (implemented + tested):**

- **Real MCP client transport.** `StdioUpstream` launches a real MCP server and speaks JSON-RPC over stdio via the official `mcp` SDK; tools are discovered live and calls are proxied to the real server. Exercised against the bundled example server (offline, in CI) and against the official `npx` filesystem server (opt-in test, and in the demo above).
- **Real MCP server transport.** `mcp_router.mcp_server` runs as a real MCP server over stdio (`find_tools` + `call_tool`); a real SDK `Client` drives the full loop in tests — including a launched-subprocess run that mirrors how Claude Desktop connects.
- **Real semantic embedder by default** (`sentence-transformers`, `all-MiniLM-L6-v2`), with the offline hashing embedder as the deterministic fallback.
- Config-driven registry (YAML/JSON), NumPy cosine top-k retrieval, upstream routing, HTTP JSON-RPC surface, Docker image.

**Not yet exercised by an external host:** connecting the *actual* Claude Desktop app is documented (config snippet above) and the stdio server it would launch is verified with the SDK's own client, but the end-to-end run *inside the Claude Desktop UI* has not been performed here.

**Deferred (next milestones):**

- **Approximate vector index** (FAISS/hnswlib) for very large tool catalogues — exact search is the right choice for tens–hundreds of tools.
- **Reranking / hybrid retrieval** (combine lexical + semantic scores).
- **SSE / streamable-HTTP** MCP transports (only stdio upstreams today).
- **Resources / prompts** passthrough and **caching** of query→tool-set results.

## License

MIT — see [LICENSE](LICENSE).
