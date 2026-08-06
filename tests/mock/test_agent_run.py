"""MS-01 → MS-09: Agent.run() event loop tests using mock streams.

These tests cover the core event loop in ``Agent.run()`` — event iteration,
tool call handling, error propagation, and forward-compatibility.
No phi binary required.
"""

from __future__ import annotations

import pytest

from phi_agent.tool import ToolOutput, tool

from tests.mock.conftest import MockStream, patch_agent


# ── MS-01: 收到 done ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_01_receives_done(mock_stream: MockStream) -> None:
    """Agent.run() yields events and stops on done."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        {"type": "event", "runtimeEventType": "textDelta", "text": "hello"},
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        events = [e async for e in agent.run("test query")]

    assert len(events) == 1
    assert events[0].type == "textDelta"
    assert events[0].text == "hello"


# ── MS-02: 收到 error ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_02_receives_error(mock_stream: MockStream) -> None:
    """Agent.run() raises RuntimeError when the server sends an error."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        {
            "type": "error",
            "code": "INVALID_REQUEST",
            "message": "bad request",
        },
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        with pytest.raises(RuntimeError, match=r"phi error.*INVALID_REQUEST.*bad request"):
            async for _ in agent.run("test query"):
                pass


# ── MS-03: 收到 tool_call + 回复 tool_result ─────────────────────────────


@pytest.mark.asyncio
async def test_ms_03_tool_call_and_result(mock_stream: MockStream) -> None:
    """Agent handles a tool_call by executing the tool and sending tool_result."""

    @tool
    async def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}"

    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        # tool registration ack
        {"type": "tool_registered", "ok": True},
        # then the run response: tool_call → (agent sends tool_result) → done
        {
            "type": "tool_call",
            "call_id": "call-1",
            "name": "greet",
            "args": {"name": "World"},
        },
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        agent.register(greet)
        events = [e async for e in agent.run("say hello")]

    # No events other than tool_call handling (tool_call is consumed, not yielded)
    assert len(events) == 0

    # Verify agent sent the correct messages
    sent = mock_stream.sent_messages
    # First: register_tool
    assert sent[0]["type"] == "register_tool"
    assert sent[0]["name"] == "greet"
    # Second: run
    assert sent[1]["type"] == "run"
    assert sent[1]["query"] == "say hello"
    # Third: tool_result
    assert sent[2]["type"] == "tool_result"
    assert sent[2]["call_id"] == "call-1"
    assert sent[2]["summary"] == "Hello, World"
    assert sent[2]["raw"] is None


# ── MS-04: tool_call 缺 call_id ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_04_tool_call_missing_call_id(mock_stream: MockStream) -> None:
    """Agent does not KeyError when tool_call lacks a call_id field."""

    @tool
    async def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}"

    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        {"type": "tool_registered", "ok": True},
        # tool_call with no "call_id" key at all
        {"type": "tool_call", "name": "greet", "args": {"name": "X"}},
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        agent.register(greet)
        events = [e async for e in agent.run("test")]

    assert len(events) == 0
    # Should not have crashed — tool_result uses empty string for call_id
    tool_results = [m for m in mock_stream.sent_messages if m["type"] == "tool_result"]
    assert len(tool_results) == 1
    # call_id defaults to "" when missing (msg.get("call_id", ""))
    assert tool_results[0]["call_id"] == ""


# ── MS-05: tool_call 未知工具名 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_05_tool_call_unknown_tool(mock_stream: MockStream) -> None:
    """Agent returns 'unknown tool' summary when tool not registered."""

    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        # no tool registration needed — agent has no tools
        {
            "type": "tool_call",
            "call_id": "call-1",
            "name": "nonexistent",
            "args": {},
        },
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        events = [e async for e in agent.run("test")]

    assert len(events) == 0

    tool_results = [m for m in mock_stream.sent_messages if m["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert "unknown tool" in tool_results[0]["summary"]
    assert "nonexistent" in tool_results[0]["summary"]


# ── MS-06: tool_call 工具抛异常 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_06_tool_call_tool_raises_exception(mock_stream: MockStream) -> None:
    """Agent sends error summary when tool raises, does not crash."""

    @tool
    async def broken() -> str:
        """Always fails."""
        raise ValueError("simulated failure")

    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        {"type": "tool_registered", "ok": True},
        {
            "type": "tool_call",
            "call_id": "call-1",
            "name": "broken",
            "args": {},
        },
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        agent.register(broken)
        events = [e async for e in agent.run("test")]

    assert len(events) == 0

    tool_results = [m for m in mock_stream.sent_messages if m["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["summary"].startswith("Error:")
    assert "simulated failure" in tool_results[0]["summary"]


# ── MS-07: tool_call 工具返回 ToolOutput ─────────────────────────────────


@pytest.mark.asyncio
async def test_ms_07_tool_returns_ToolOutput(mock_stream: MockStream) -> None:
    """ToolOutput.raw is correctly passed through to tool_result."""

    @tool
    async def structured() -> ToolOutput:
        """Returns structured data."""
        return ToolOutput(summary="done", raw={"count": 42, "items": ["a", "b"]})

    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        {"type": "tool_registered", "ok": True},
        {
            "type": "tool_call",
            "call_id": "call-1",
            "name": "structured",
            "args": {},
        },
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        agent.register(structured)
        events = [e async for e in agent.run("test")]

    assert len(events) == 0

    tool_results = [m for m in mock_stream.sent_messages if m["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["summary"] == "done"
    assert tool_results[0]["raw"] == {"count": 42, "items": ["a", "b"]}


# ── MS-08: 未知消息类型 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_08_unknown_message_type_is_ignored(mock_stream: MockStream) -> None:
    """Unknown message types are silently ignored (forward-compat)."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        {"type": "future_msg", "payload": "unexpected"},
        {"type": "event", "runtimeEventType": "textDelta", "text": "ok"},
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        events = [e async for e in agent.run("test")]

    # future_msg is ignored, only the textDelta is yielded
    assert len(events) == 1
    assert events[0].type == "textDelta"
    assert events[0].text == "ok"


# ── MS-09: 多条事件后 done ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_09_multiple_events_then_done(mock_stream: MockStream) -> None:
    """All events are yielded, and iteration stops at done."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        {
            "type": "event",
            "runtimeEventType": "thoughtDelta",
            "text": "thinking...",
        },
        {"type": "event", "runtimeEventType": "textDelta", "text": "part1"},
        {"type": "event", "runtimeEventType": "textDelta", "text": "part2"},
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        events = [e async for e in agent.run("test")]

    assert len(events) == 3
    assert events[0].type == "thoughtDelta"
    assert events[0].text == "thinking..."
    assert events[1].type == "textDelta"
    assert events[1].text == "part1"
    assert events[2].type == "textDelta"
    assert events[2].text == "part2"


# ── MS-10: 完整 RunConfig 参数 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_10_full_run_config_sent(mock_stream: MockStream) -> None:
    """Agent.run() sends all configured RunConfig fields."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.2.7"},
        {"type": "event", "runtimeEventType": "textDelta", "text": "ok"},
        {"type": "done"},
    )

    with patch_agent(
        mock_stream,
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        enable_thinking=False,
        thinking_effort="low",
        thinking_budget=16000,
        max_tool_calls_per_turn=10,
        max_consecutive_failures=3,
        max_turns=5,
    ) as agent:
        events = [e async for e in agent.run("test")]
        assert len(events) == 1

    # Check the run message includes all config fields
    run_msg = [m for m in mock_stream.sent_messages if m["type"] == "run"][0]
    cfg = run_msg["config"]
    assert cfg["model"] == "gpt-4o-mini"
    assert cfg["api_key"] == "sk-test"
    assert cfg["base_url"] == "https://api.example.com/v1"
    assert cfg["enable_thinking"] is False
    assert cfg["thinking_effort"] == "low"
    assert cfg["thinking_budget"] == 16000
    assert cfg["max_tool_calls_per_turn"] == 10
    assert cfg["max_consecutive_failures"] == 3
    assert cfg["max_turns"] == 5


# ── MS-11: RunConfig 无 None 字段 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_11_run_config_omits_none_fields(mock_stream: MockStream) -> None:
    """Agent.run() does not include None-valued optional config fields."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.2.7"},
        {"type": "event", "runtimeEventType": "textDelta", "text": "ok"},
        {"type": "done"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        events = [e async for e in agent.run("test")]
        assert len(events) == 1

    run_msg = [m for m in mock_stream.sent_messages if m["type"] == "run"][0]
    cfg = run_msg["config"]
    # These should be present (always sent)
    assert "model" in cfg
    assert "enable_thinking" in cfg
    assert "thinking_effort" in cfg
    # These should NOT be present (None values omitted)
    assert "thinking_budget" not in cfg
    assert "max_tool_calls_per_turn" not in cfg
    assert "max_consecutive_failures" not in cfg
    assert "max_turns" not in cfg
    assert "api_key" not in cfg
    assert "base_url" not in cfg
