"""FastAPI app exposing the gateway over an MCP-shaped JSON-RPC interface.

MCP itself is JSON-RPC 2.0. To stay recognizable to MCP tooling we accept the
same envelope and the same two methods a client cares about here:

* ``tools/list``  -> returns tool definitions. Our extension: an optional
  ``query`` (and ``k``) in ``params`` makes it return only the top-k relevant
  tools instead of all of them. That is the whole point of the gateway.
* ``tools/call``  -> ``params`` = ``{"name", "arguments"}``; routed to the
  upstream that owns the tool.

The endpoint is a single ``POST /`` (JSON-RPC is transport-agnostic and usually
posts to one URL). We add ``GET /healthz`` for container/orchestrator probes.

The web layer is deliberately thin: it validates the JSON-RPC envelope and
delegates all logic to :class:`ToolRegistry`.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .registry import ToolRegistry

# Standard JSON-RPC 2.0 error codes we use.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603

# Default number of tools to expose per query. Overridable per-request via
# params.k, or globally via the MCP_ROUTER_DEFAULT_K env var.
DEFAULT_K = int(os.environ.get("MCP_ROUTER_DEFAULT_K", "5"))


def _rpc_error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _rpc_result(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def handle_rpc(registry: ToolRegistry, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process one decoded JSON-RPC request and return the response dict.

    Kept as a plain function (no FastAPI types) so the whole dispatch path is
    unit-testable without spinning up an HTTP client.
    """
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        return _rpc_error(payload.get("id") if isinstance(payload, dict) else None,
                          _INVALID_REQUEST, "not a valid JSON-RPC 2.0 request")

    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        return _rpc_error(req_id, _INVALID_PARAMS, "params must be an object")

    if method == "tools/list":
        query = params.get("query")
        k = int(params.get("k", DEFAULT_K))
        tools = registry.list_tools(query=query, k=k)
        return _rpc_result(req_id, {"tools": [t.to_mcp() for t in tools]})

    if method == "tools/call":
        name = params.get("name")
        if not name:
            return _rpc_error(req_id, _INVALID_PARAMS, "tools/call requires 'name'")
        arguments = params.get("arguments") or {}
        try:
            result = registry.call_tool(name, arguments)
        except KeyError as exc:
            return _rpc_error(req_id, _INVALID_PARAMS, str(exc))
        return _rpc_result(req_id, result)

    return _rpc_error(req_id, _METHOD_NOT_FOUND, f"unknown method: {method!r}")


def create_app(registry: ToolRegistry) -> FastAPI:
    """Build a FastAPI app bound to an already-constructed registry.

    Taking the registry as an argument (dependency injection) keeps the app
    testable: a test can pass a registry built from an in-memory config and a
    deterministic embedder.
    """
    app = FastAPI(
        title="MCP Tool-Retrieval Gateway",
        description="Exposes only the top-k semantically relevant MCP tools per query.",
    )

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"status": "ok", "tools": registry.tool_count()}

    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        # Operational metrics, incl. the v0.3 query-cache hit/miss counters.
        return registry.stats()

    @app.post("/")
    async def rpc(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(_rpc_error(None, _PARSE_ERROR, "invalid JSON body"))
        return JSONResponse(handle_rpc(registry, payload))

    return app


def app_from_env() -> FastAPI:
    """Construct the app from the ``MCP_ROUTER_CONFIG`` env var.

    Used by ``uvicorn mcp_router.server:app_from_env --factory`` and by the
    Docker image's default command.
    """
    config_path = os.environ.get("MCP_ROUTER_CONFIG", "config.example.yaml")
    registry = ToolRegistry.from_config_file(config_path)
    return create_app(registry)
