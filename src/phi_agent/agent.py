"""Agent — the main entry point for Python users."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from .events import Event
from .process import PhiProcess
from .tool import RegisteredTool

_log = logging.getLogger("phi-agent")


class Agent:
    """A phi-agent instance backed by the Rust runtime.

    Usage::

        agent = Agent(model="gpt-4o")
        agent.register(my_tool)

        async for event in agent.run("What is the weather?"):
            print(event)
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        enable_thinking: bool = True,
        thinking_effort: str = "medium",
        phi_path: str | None = None,
    ) -> None:
        # Resolve config (env vars are fallbacks, not overrides)
        self._model = model or os.environ.get("LLM_MODEL", "gpt-4o")
        self._api_key = api_key or os.environ.get("LLM_API_KEY")
        self._base_url = base_url or os.environ.get("LLM_BASE_URL")
        self._enable_thinking = enable_thinking
        self._thinking_effort = thinking_effort
        self._phi_path = phi_path

        # Registered tools (name → RegisteredTool)
        self._tools: dict[str, RegisteredTool] = {}

        # Subprocess manager
        self._proc: PhiProcess | None = None

    # ── Tool registration ──────────────────────────────────────────────

    def register(self, tool: RegisteredTool) -> None:
        """Register a tool (function decorated with ``@tool``)."""
        self._tools[tool.name] = tool

    # ── Run ────────────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        *,
        session_id: str | None = None,
    ) -> AsyncIterator[Event]:
        """Execute one agent turn, yielding events as they arrive.

        Spawns ``phi serve`` on first call and reuses the process for
        subsequent turns.

        Yields:
            :class:`Event` objects — one per protocol event.
        """
        if self._proc is None:
            self._proc = PhiProcess(phi_path=self._phi_path)
            await self._proc.start()
            await self._proc.handshake()
            await self._register_tools()

        stream = self._proc.stream

        # Send run request
        config: dict[str, Any] = {
            "model": self._model,
            "enable_thinking": self._enable_thinking,
            "thinking_effort": self._thinking_effort,
        }
        if self._api_key:
            config["api_key"] = self._api_key
        if self._base_url:
            config["base_url"] = self._base_url

        await stream.send({
            "type": "run",
            "session_id": session_id or "",
            "query": query,
            "config": config,
        })

        # Stream events until "done"
        while True:
            msg = await stream.recv()
            msg_type = msg.get("type", "")

            if msg_type == "event":
                yield Event(type=msg.get("runtimeEventType", "unknown"), data=msg)

            elif msg_type == "tool_call":
                # Rust side wants to call a Python tool
                await self._handle_tool_call(msg, stream)

            elif msg_type == "done":
                return

            elif msg_type == "error":
                raise RuntimeError(
                    f"phi error [{msg.get('code', 'UNKNOWN')}]: {msg.get('message', '')}"
                )

            else:
                # Ignore unknown message types (forward-compat)
                pass

    # ── Internals ──────────────────────────────────────────────────────

    async def _register_tools(self) -> None:
        """Send all registered tool definitions to the Rust runtime."""
        for tool in self._tools.values():
            try:
                await self._proc.stream.send({  # type: ignore[union-attr]
                    "type": "register_tool",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
                ack = await self._proc.stream.recv()  # type: ignore[union-attr]
                if ack.get("type") != "tool_registered" or not ack.get("ok"):
                    raise RuntimeError(f"tool registration failed: {tool.name} → {ack}")
            except Exception:
                _log.exception(f"failed to register tool: {tool.name}")
                raise

    async def _handle_tool_call(
        self,
        msg: dict[str, Any],
        stream: Any,
    ) -> None:
        """Execute a Python-side tool and send the result back."""
        call_id: str = msg.get("call_id", "")
        tool_name: str = msg.get("name", "")
        args: dict[str, Any] = msg.get("args", {})

        tool = self._tools.get(tool_name)
        if tool is None:
            await stream.send({
                "type": "tool_result",
                "call_id": call_id,
                "summary": f"Error: unknown tool '{tool_name}'",
                "raw": None,
                "control_flow": "break",
            })
            return

        try:
            result = await tool.func(**args)
            if isinstance(result, str):
                summary = result
                raw = None
            else:
                from .tool import ToolOutput
                if isinstance(result, ToolOutput):
                    summary = result.summary
                    raw = result.raw
                else:
                    summary = str(result)
                    raw = None

            await stream.send({
                "type": "tool_result",
                "call_id": call_id,
                "summary": summary,
                "raw": raw,
                "control_flow": "break",
            })
        except Exception as exc:
            await stream.send({
                "type": "tool_result",
                "call_id": call_id,
                "summary": f"Error: {exc}",
                "raw": None,
                "control_flow": "break",
            })

    # ── Cancel ──────────────────────────────────────────────────────────

    async def cancel(self) -> None:
        """Cancel the currently running turn.

        Sends a cancel message to phi serve.  The next event yielded by
        :meth:`run` will be ``RunCancelled``, followed by ``done``.
        """
        if self._proc is not None:
            await self._proc.stream.send({"type": "cancel"})

    # ── Cleanup ────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Terminate the phi subprocess."""
        if self._proc is not None:
            await self._proc.close()
            self._proc = None
