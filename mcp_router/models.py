"""Core data structures shared across the gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ToolDef:
    """A single MCP tool definition plus the upstream that owns it.

    Fields mirror the MCP ``tools/list`` schema (``name``, ``description``,
    ``inputSchema``) with one addition: ``upstream``, the id of the server this
    tool came from. The gateway needs that so ``tools/call`` can route an
    invocation back to the correct upstream.
    """

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    upstream: str = "inline"

    def embedding_text(self) -> str:
        """The text used to represent this tool in vector space.

        We combine the name, description, and the tool's parameter names. The
        parameter names matter: a query like "convert currency amount" should
        match a tool whose schema has an ``amount`` field even if the prose
        wording differs.
        """
        parts = [self.name.replace("_", " "), self.description]
        props = self.input_schema.get("properties")
        if isinstance(props, dict):
            parts.extend(props.keys())
        return " ".join(p for p in parts if p)

    def to_mcp(self) -> Dict[str, Any]:
        """Render as an MCP-shaped tool object for ``tools/list`` responses."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
