"""MS-10 → MS-12: _register_tools tests using mock streams.

Tests the tool registration protocol: successful registration,
failed ack detection, and phi crash during registration.
No phi binary required.
"""

from __future__ import annotations

import pytest

from phi_agent.protocol import ProtocolError
from phi_agent.tool import tool

from tests.mock.conftest import MockStream, patch_agent


# ── MS-10: 注册成功 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_10_register_tool_success(mock_stream: MockStream) -> None:
    """Tool registration with ok=true does not raise an exception."""

    @tool
    async def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}"

    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        # tool registration ack — success
        {"type": "tool_registered", "ok": True},
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        agent.register(greet)
        # Should not raise
        events = [e async for e in agent.run("test")]

    assert len(events) == 0

    # Verify register_tool was sent before run
    sent = mock_stream.sent_messages
    assert sent[0]["type"] == "register_tool"
    assert sent[0]["name"] == "greet"
    assert sent[0]["description"] == "Say hello."
    assert "properties" in sent[0]["parameters"]


# ── MS-11: 注册失败 ack ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_11_register_tool_failed_ack(mock_stream: MockStream) -> None:
    """Tool registration with ok=false raises RuntimeError (not silent success)."""

    @tool
    async def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}"

    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        # tool registration ack — failure
        {"type": "tool_registered", "ok": False},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        agent.register(greet)
        with pytest.raises(RuntimeError, match="tool registration failed"):
            async for _ in agent.run("test"):
                pass


# ── MS-12: 注册中 phi crash ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_12_register_tool_phi_crash(mock_stream: MockStream) -> None:
    """ProtocolError during registration is propagated, not silently swallowed."""

    @tool
    async def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}"

    # Simulate: hello succeeds, but recv() fails during registration
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        # Inject ProtocolError to simulate phi crash during recv()
        ProtocolError("phi process closed stdout unexpectedly"),
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        agent.register(greet)
        # ProtocolError propagates — _register_tools logs and re-raises
        with pytest.raises(ProtocolError, match="phi process closed stdout unexpectedly"):
            async for _ in agent.run("test"):
                pass

    # Verify register_tool was sent before the crash
    sent = mock_stream.sent_messages
    assert sent[0]["type"] == "register_tool"
    assert sent[0]["name"] == "greet"
