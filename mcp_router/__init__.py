"""MCP Tool-Retrieval Gateway.

A self-hostable MCP proxy that embeds all upstream tool definitions and, per
query, exposes only the top-k semantically relevant tools - cutting the context
bloat that comes from loading every tool from every connected MCP server.
"""

from .cache import QueryCache, normalize_query
from .config import CacheConfig, GatewayConfig, load_config, parse_config
from .embedder import Embedder, HashingEmbedder, get_embedder
from .models import ToolDef
from .registry import ToolRegistry
from .retrieval import Retriever
from .upstream import MockUpstream, StdioUpstream, Upstream
from .vectorstore import VectorStore

__version__ = "0.3.0"

__all__ = [
    "CacheConfig",
    "GatewayConfig",
    "load_config",
    "parse_config",
    "Embedder",
    "HashingEmbedder",
    "get_embedder",
    "ToolDef",
    "ToolRegistry",
    "QueryCache",
    "normalize_query",
    "Retriever",
    "Upstream",
    "MockUpstream",
    "StdioUpstream",
    "VectorStore",
    "__version__",
]
