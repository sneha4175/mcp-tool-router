"""Upstream transports: where ``tools/call`` actually gets executed.

The gateway routes a tool call to whichever upstream owns that tool. We isolate
that behind a tiny :class:`Upstream` protocol so the rest of the system does not
care whether a call is served by an in-process mock or a real MCP server over
stdio.

* :class:`MockUpstream` - deterministic, offline. It echoes the call back as a
  structured result. This is what the MVP and the tests use, and it is enough to
  prove the *routing* is correct (the right upstream receives the right call).
* Real MCP stdio transport (spawning a server subprocess and speaking JSON-RPC
  over stdin/stdout via the official ``mcp`` SDK) is the documented next
  milestone. We deliberately do not ship a half-working version - see the README
  roadmap.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol


class Upstream(Protocol):
    """Anything that can execute a tool call for one upstream server."""

    name: str

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ``tool_name`` with ``arguments`` and return an MCP result."""
        ...


class MockUpstream:
    """An offline upstream that echoes calls back as a text content block.

    The return value is shaped like an MCP ``tools/call`` result (a ``content``
    list of typed blocks) so downstream code sees the same structure it would
    from a real server. The echoed payload includes the upstream name, which is
    what lets a test assert the call was routed to the correct server.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"[mock:{self.name}] executed {tool_name} "
                        f"with arguments {arguments}"
                    ),
                }
            ],
            "isError": False,
            # Non-standard echo fields, handy for tests and debugging. A real
            # server would not include these.
            "_upstream": self.name,
            "_tool": tool_name,
            "_arguments": arguments,
        }
