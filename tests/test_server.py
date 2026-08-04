"""Tests for the MCP-shaped JSON-RPC interface.

These assert the two behaviours that make the gateway worth having:
1. ``tools/list`` with a query returns only the top-k tools, not all of them.
2. ``tools/call`` is routed to the upstream that actually owns the tool.
"""

from __future__ import annotations

from mcp_router.registry import ToolRegistry
from mcp_router.server import create_app, handle_rpc

from .conftest import TOTAL_TOOLS


def _rpc(registry, method, params=None, req_id=1):
    return handle_rpc(
        registry,
        {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}},
    )


def test_tools_list_without_query_returns_all(registry: ToolRegistry):
    resp = _rpc(registry, "tools/list")
    assert len(resp["result"]["tools"]) == TOTAL_TOOLS


def test_tools_list_with_query_returns_only_top_k(registry: ToolRegistry):
    # The whole point: N total tools reduced to k exposed for a query.
    resp = _rpc(registry, "tools/list", {"query": "convert currency", "k": 2})
    tools = resp["result"]["tools"]
    assert len(tools) == 2
    assert len(tools) < TOTAL_TOOLS
    assert tools[0]["name"] == "convert_currency"


def test_tools_list_response_is_mcp_shaped(registry: ToolRegistry):
    tool = _rpc(registry, "tools/list", {"query": "read a file", "k": 1})["result"]["tools"][0]
    assert set(tool.keys()) == {"name", "description", "inputSchema"}


def test_tools_call_routes_to_owning_upstream(registry: ToolRegistry):
    resp = _rpc(registry, "tools/call", {"name": "send_email", "arguments": {"to": "x@y.com"}})
    result = resp["result"]
    # send_email lives on the "files" upstream in the sample config, so that is
    # where the call must have been routed.
    assert result["_upstream"] == "files"
    assert result["_tool"] == "send_email"
    assert result["_arguments"] == {"to": "x@y.com"}
    assert result["isError"] is False


def test_tools_call_different_tool_routes_to_finance(registry: ToolRegistry):
    resp = _rpc(registry, "tools/call", {"name": "get_stock_price", "arguments": {"symbol": "ACME"}})
    assert resp["result"]["_upstream"] == "finance"


def test_tools_call_unknown_tool_is_invalid_params(registry: ToolRegistry):
    resp = _rpc(registry, "tools/call", {"name": "does_not_exist", "arguments": {}})
    assert resp["error"]["code"] == -32602


def test_unknown_method_returns_method_not_found(registry: ToolRegistry):
    resp = _rpc(registry, "resources/list")
    assert resp["error"]["code"] == -32601


def test_bad_envelope_is_invalid_request(registry: ToolRegistry):
    resp = handle_rpc(registry, {"id": 1, "method": "tools/list"})  # missing jsonrpc
    assert resp["error"]["code"] == -32600


def test_http_endpoint_end_to_end(registry: ToolRegistry):
    # Exercise the real FastAPI path with the test client.
    from fastapi.testclient import TestClient

    client = TestClient(create_app(registry))

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["tools"] == TOTAL_TOOLS

    resp = client.post(
        "/",
        json={"jsonrpc": "2.0", "id": 7, "method": "tools/list",
              "params": {"query": "weather forecast", "k": 1}},
    )
    body = resp.json()
    assert body["id"] == 7
    assert len(body["result"]["tools"]) == 1
    assert body["result"]["tools"][0]["name"] == "get_forecast"
