"""Upstream transports: where tool discovery and ``tools/call`` actually happen.

The gateway fronts one or more upstream MCP servers. Each upstream is responsible
for two things behind a small :class:`Upstream` protocol, so the rest of the
system never cares *how* a server is reached:

* ``list_tools()`` - the tools this upstream offers (used at startup to build the
  retrieval index).
* ``call(name, arguments)`` - execute a tool and return an MCP-shaped result.

Two implementations ship:

* :class:`MockUpstream` - deterministic, offline. Its tools come from the config
  file and its calls are echoed back. Ideal for tests, demos, and running with no
  external processes. It proves the *routing* logic without a real subprocess.
* :class:`StdioUpstream` - a **real** MCP client. It launches an MCP server as a
  subprocess and speaks JSON-RPC over stdio using the official ``mcp`` SDK
  (``stdio_client`` + ``ClientSession``). Tools are discovered live from the
  server and calls are proxied to it. This is the real transport that turns the
  gateway from a mock into a working proxy.

Bridging sync and async
-----------------------
The SDK's client is asyncio-based and a session must stay open for the lifetime
of the connection (both ``stdio_client`` and ``ClientSession`` are async context
managers). The rest of this codebase - the registry and the FastAPI dispatch -
is synchronous. Rather than make everything async, :class:`StdioUpstream` runs a
dedicated event loop on a background thread, opens the session there once, and
keeps it alive. Synchronous ``list_tools``/``call`` submit coroutines to that
loop and block for the result. This keeps a single long-lived subprocess per
upstream (how MCP servers are meant to be used) while presenting a plain
synchronous API to the caller.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Protocol, Sequence

from .models import ToolDef


class Upstream(Protocol):
    """Anything that can list and execute tools for one upstream server."""

    name: str

    def list_tools(self) -> List[ToolDef]:
        """Return the tools this upstream exposes, tagged with its own name."""
        ...

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ``tool_name`` with ``arguments`` and return an MCP result."""
        ...

    def close(self) -> None:
        """Release any resources (subprocess, thread). Safe to call twice."""
        ...


class MockUpstream:
    """An offline upstream: tools come from config, calls are echoed back.

    The return value is shaped like an MCP ``tools/call`` result (a ``content``
    list of typed blocks) so downstream code sees the same structure it would get
    from a real server. The echoed ``_upstream``/``_tool`` fields let a test
    assert the call was routed to the correct server.
    """

    def __init__(self, name: str, tools: Sequence[ToolDef] = ()) -> None:
        self.name = name
        self._tools = list(tools)

    def list_tools(self) -> List[ToolDef]:
        return list(self._tools)

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
            # Non-standard echo fields, handy for tests/debugging. A real server
            # would not include these.
            "_upstream": self.name,
            "_tool": tool_name,
            "_arguments": arguments,
        }

    def close(self) -> None:  # nothing to release
        return None


def _content_to_dict(block: Any) -> Dict[str, Any]:
    """Best-effort convert an SDK content block to a plain JSON-able dict."""
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(block, dict):
        return block
    return {"type": "text", "text": str(block)}


class StdioUpstream:
    """A real MCP client over stdio, backed by a long-lived subprocess.

    Launches ``command`` + ``args`` as an MCP server and connects to it with the
    official SDK. Tool discovery and calls are proxied to the live server.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: Sequence[str] = (),
        env: Dict[str, str] | None = None,
        *,
        connect_timeout: float = 60.0,
        call_timeout: float = 60.0,
    ) -> None:
        if not command:
            raise ValueError(f"stdio upstream {name!r} needs a 'command'")
        self.name = name
        self._command = command
        self._args = list(args)
        self._env = dict(env) if env else None
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stop: asyncio.Event | None = None
        self._closed = False

        self._start()

    # ---- lifecycle -----------------------------------------------------------

    def _start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"mcp-upstream-{self.name}", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=self._connect_timeout):
            raise TimeoutError(
                f"stdio upstream {self.name!r} did not connect within "
                f"{self._connect_timeout}s (command: {self._command} {self._args})"
            )
        if self._error is not None:
            raise RuntimeError(
                f"stdio upstream {self.name!r} failed to start: {self._error}"
            ) from self._error

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:  # surfaced to the constructor via _ready
            self._error = exc
            self._ready.set()

    async def _serve(self) -> None:
        # Imported here so a config that uses no stdio upstreams never pays the
        # SDK import cost, and a missing SDK fails only when actually needed.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        params = StdioServerParameters(
            command=self._command, args=self._args, env=self._env
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()  # keep the session open until close()
        except Exception as exc:
            self._error = exc
            self._ready.set()

    def _submit(self, coro) -> Any:
        if self._loop is None or self._session is None:
            raise RuntimeError(f"stdio upstream {self.name!r} is not connected")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self._call_timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---- MCP operations ------------------------------------------------------

    def list_tools(self) -> List[ToolDef]:
        result = self._submit(self._session.list_tools())
        tools: List[ToolDef] = []
        for tool in result.tools:
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            tools.append(
                ToolDef(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=schema or {},
                    upstream=self.name,
                )
            )
        return tools

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = self._submit(self._session.call_tool(tool_name, arguments or {}))
        content = [_content_to_dict(block) for block in (result.content or [])]
        return {
            "content": content,
            "isError": bool(getattr(result, "is_error", False)),
            "_upstream": self.name,
            "_tool": tool_name,
            "_arguments": arguments,
        }
