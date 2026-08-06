"""Shared pytest fixtures.

Every test builds on a deterministic HashingEmbedder and a small in-memory
config, so the whole suite runs offline and gives identical results on every
machine and every run.
"""

from __future__ import annotations

import os

# The default embedder is a real model as of v0.2; force the offline hashing
# embedder for the whole suite so tests never download weights or hit a network.
# Individual tests still inject their own embedder, but this covers any code path
# that falls back to get_embedder().
os.environ.setdefault("MCP_ROUTER_EMBEDDER", "hashing")

import pytest

from mcp_router.config import parse_config
from mcp_router.embedder import HashingEmbedder
from mcp_router.registry import ToolRegistry

# A compact multi-upstream config used across the suite. Tools are chosen so
# that distinct queries have an unambiguous best match.
SAMPLE_CONFIG = {
    "servers": [
        {
            "name": "weather",
            "transport": "mock",
            "tools": [
                {
                    "name": "get_current_weather",
                    "description": "Get the current temperature and conditions for a city.",
                    "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
                {
                    "name": "get_forecast",
                    "description": "Get a multi-day weather forecast for a location.",
                    "inputSchema": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            ],
        },
        {
            "name": "finance",
            "transport": "mock",
            "tools": [
                {
                    "name": "convert_currency",
                    "description": "Convert a monetary amount from one currency to another.",
                    "inputSchema": {"type": "object", "properties": {"amount": {"type": "number"}}},
                },
                {
                    "name": "get_stock_price",
                    "description": "Look up the latest share price for a stock ticker symbol.",
                    "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}},
                },
            ],
        },
        {
            "name": "files",
            "transport": "mock",
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read the contents of a file from disk.",
                    "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
                {
                    "name": "send_email",
                    "description": "Send an email message to one or more recipients.",
                    "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}}},
                },
            ],
        },
    ]
}

TOTAL_TOOLS = 6  # keep in sync with SAMPLE_CONFIG above


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture
def registry(embedder: HashingEmbedder) -> ToolRegistry:
    return ToolRegistry(parse_config(SAMPLE_CONFIG), embedder=embedder)
