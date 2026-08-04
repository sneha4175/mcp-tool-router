"""MCP Tool-Retrieval Gateway.

A self-hostable MCP proxy that embeds all upstream tool definitions and, per
query, exposes only the top-k semantically relevant tools - cutting the context
bloat that comes from loading every tool from every connected MCP server.
"""

from .config import GatewayConfig, load_config, parse_config
from .embedder import Embedder, HashingEmbedder, get_embedder
from .models import ToolDef
from .registry import ToolRegistry
from .retrieval import Retriever
from .vectorstore import VectorStore

__version__ = "0.1.0"

__all__ = [
    "GatewayConfig",
    "load_config",
    "parse_config",
    "Embedder",
    "HashingEmbedder",
    "get_embedder",
    "ToolDef",
    "ToolRegistry",
    "Retriever",
    "VectorStore",
    "__version__",
]
