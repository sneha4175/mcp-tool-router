"""Loading and validating the gateway configuration.

The config declares which upstream MCP servers the gateway fronts and, for the
offline/mock path, the tools each one exposes. Keeping this in a YAML/JSON file
(rather than hard-coded) is what makes the gateway self-hostable: an operator
points it at their own servers without touching code.

Config shape (YAML shown; JSON with the same keys also works)::

    servers:
      - name: weather              # unique upstream id
        transport: mock            # "mock" (offline) or "stdio" (real MCP)
        tools:                     # inline tool defs (mock transport)
          - name: get_forecast
            description: Get the weather forecast for a location.
            inputSchema:
              type: object
              properties:
                location: {type: string}
      - name: filesystem
        transport: stdio           # real MCP server launched as a subprocess
        command: npx               # executable
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
        env: {}                    # optional extra environment variables

Both transports are fully wired: ``mock`` serves inline tools offline, ``stdio``
launches a real MCP server and speaks the protocol to it via the ``mcp`` SDK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .models import ToolDef


@dataclass
class ServerConfig:
    """One upstream server entry from the config file."""

    name: str
    transport: str = "mock"
    # stdio transport: the executable and its arguments.
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    # mock transport: inline tool definitions.
    tools: List[ToolDef] = field(default_factory=list)


@dataclass
class CacheConfig:
    """Query-result cache settings (v0.3).

    All optional with sensible defaults, so existing configs keep working with
    the cache silently on. Operators tune it under a top-level ``cache:`` block::

        cache:
          enabled: true
          ttl_seconds: 300      # how long a cached tool set stays fresh
          max_entries: 512      # LRU cap on distinct cached queries
    """

    enabled: bool = True
    ttl_seconds: float = 300.0
    max_entries: int = 512


@dataclass
class RetrievalConfig:
    """Retrieval-mode settings (v0.4 hybrid retrieval).

    Hybrid retrieval blends the semantic (embedding cosine) score with a lexical
    token-overlap score so an exact tool-name/keyword match surfaces even when
    the embedding similarity is only moderate. Operators tune it under a
    top-level ``retrieval:`` block::

        retrieval:
          hybrid: true      # blend semantic + lexical (default). false = pure semantic
          alpha: 0.5        # weight: final = alpha*semantic + (1-alpha)*lexical

    Defaults keep hybrid **on**: it strictly helps exact-match queries and only
    reranks within the same catalogue, so existing configs improve for free.
    Set ``alpha: 1.0`` to reproduce the v0.3 pure-semantic behaviour.
    """

    hybrid: bool = True
    alpha: float = 0.5


@dataclass
class GatewayConfig:
    """The whole parsed configuration."""

    servers: List[ServerConfig] = field(default_factory=list)
    cache: CacheConfig = field(default_factory=CacheConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def all_tools(self) -> List[ToolDef]:
        tools: List[ToolDef] = []
        for server in self.servers:
            tools.extend(server.tools)
        return tools


def _parse_tool(raw: Dict[str, Any], upstream: str) -> ToolDef:
    if "name" not in raw:
        raise ValueError(f"tool in server {upstream!r} is missing 'name'")
    return ToolDef(
        name=raw["name"],
        description=raw.get("description", ""),
        # Accept both the MCP-native "inputSchema" and a snake_case alias.
        input_schema=raw.get("inputSchema") or raw.get("input_schema") or {},
        upstream=upstream,
    )


def parse_config(data: Dict[str, Any]) -> GatewayConfig:
    """Build a :class:`GatewayConfig` from an already-decoded dict."""
    servers: List[ServerConfig] = []
    seen = set()
    for raw_server in data.get("servers", []):
        name = raw_server.get("name")
        if not name:
            raise ValueError("every server entry needs a 'name'")
        if name in seen:
            raise ValueError(f"duplicate server name: {name!r}")
        seen.add(name)

        tools = [_parse_tool(t, upstream=name) for t in raw_server.get("tools", [])]

        # ``command`` may be a plain string ("npx") or, for convenience, a list
        # whose first element is the executable and the rest are arguments.
        raw_command = raw_server.get("command", "")
        args = list(raw_server.get("args", []) or [])
        if isinstance(raw_command, list):
            command = raw_command[0] if raw_command else ""
            args = list(raw_command[1:]) + args
        else:
            command = raw_command or ""

        servers.append(
            ServerConfig(
                name=name,
                transport=raw_server.get("transport", "mock"),
                command=command,
                args=args,
                env=dict(raw_server.get("env", {}) or {}),
                tools=tools,
            )
        )

    cache = _parse_cache(data.get("cache") or {})
    retrieval = _parse_retrieval(data.get("retrieval") or {})
    return GatewayConfig(servers=servers, cache=cache, retrieval=retrieval)


def _parse_cache(raw: Dict[str, Any]) -> CacheConfig:
    """Build a :class:`CacheConfig`, falling back to defaults for absent keys."""
    if not isinstance(raw, dict):
        raise ValueError("'cache' must be a mapping/object")
    defaults = CacheConfig()
    return CacheConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        ttl_seconds=float(raw.get("ttl_seconds", defaults.ttl_seconds)),
        max_entries=int(raw.get("max_entries", defaults.max_entries)),
    )


def _parse_retrieval(raw: Dict[str, Any]) -> RetrievalConfig:
    """Build a :class:`RetrievalConfig`, falling back to defaults for absent keys."""
    if not isinstance(raw, dict):
        raise ValueError("'retrieval' must be a mapping/object")
    defaults = RetrievalConfig()
    alpha = float(raw.get("alpha", defaults.alpha))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"retrieval.alpha must be in [0, 1], got {alpha!r}")
    return RetrievalConfig(
        hybrid=bool(raw.get("hybrid", defaults.hybrid)),
        alpha=alpha,
    )


def load_config(path: str | Path) -> GatewayConfig:
    """Load config from a ``.yaml``/``.yml`` or ``.json`` file.

    YAML support is optional: if the file is YAML but PyYAML is not installed we
    raise a clear error rather than silently misbehaving.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Reading YAML config requires PyYAML (`pip install pyyaml`), or "
                "use a .json config instead."
            ) from exc
        data = yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix!r}")

    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping/object")
    return parse_config(data)
