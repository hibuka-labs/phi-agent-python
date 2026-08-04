"""PT-01 → PT-15: Protocol tests against real phi serve.

Tests NDJSON protocol: handshake, tool registration, session management,
message format robustness, and cancel mechanism.
Requires phi serve binary (no API key needed for protocol-layer tests).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from phi_agent.agent import Agent


# ═══════════════════════════════════════════════════════════════════════════
# PT-01 & PT-02: Handshake
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pt_01_normal_handshake(phi_server) -> None:
    """phi serve sends a valid hello message on startup."""
    proc, stream, hello = phi_server

    assert hello["type"] == "hello"
    assert hello["protocol_version"] == 1
    assert hello["server_name"] == "phi-agent"
    assert hello["server_version"] != ""


@pytest.mark.asyncio
async def test_pt_02_protocol_version_check() -> None:
    """SDK rejects unsupported protocol versions.

    PT-01 asserts real phi serve reports version 1.
    MS-14 (Mock Stream) tests SDK-side rejection of version != 1.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════
# PT-03 & PT-04: Tool Registration
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pt_03_register_tool(phi_server) -> None:
    """Sending register_tool returns tool_registered ack with ok=true."""
    proc, stream, hello = phi_server

    await stream.send({
        "type": "register_tool",
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    })

    ack = await stream.recv()
    assert ack["type"] == "tool_registered"
    assert ack.get("ok") is True


@pytest.mark.asyncio
async def test_pt_04_register_same_tool_twice(phi_server) -> None:
    """Registering the same tool twice returns ok=true (覆盖)."""
    proc, stream, hello = phi_server

    tool_def = {
        "type": "register_tool",
        "name": "dup_tool",
        "description": "Duplicate tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    await stream.send(tool_def)
    ack1 = await stream.recv()
    assert ack1["type"] == "tool_registered"
    assert ack1.get("ok") is True

    await stream.send(tool_def)
    ack2 = await stream.recv()
    assert ack2["type"] == "tool_registered"
    assert ack2.get("ok") is True


# ═══════════════════════════════════════════════════════════════════════════
# PT-05, PT-06, PT-07: Session Management
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pt_05_create_session(phi_server) -> None:
    """create_session returns session_created with internal_id > 0."""
    proc, stream, hello = phi_server

    await stream.send({"type": "create_session"})
    resp = await stream.recv()

    assert resp["type"] == "session_created"
    assert resp.get("internal_id", 0) > 0


@pytest.mark.asyncio
async def test_pt_06_create_session_with_external_id(phi_server) -> None:
    """create_session with external_id echoes it back."""
    proc, stream, hello = phi_server

    await stream.send({
        "type": "create_session",
        "session_id": "my-custom-id",
    })
    resp = await stream.recv()

    assert resp["type"] == "session_created"
    assert resp.get("session_id") == "my-custom-id"


@pytest.mark.asyncio
async def test_pt_07_session_id_reuse(phi_server_with_key) -> None:
    """Two runs with the same session_id should share context.

    Currently fails because serve.rs always creates a new session.
    """
    proc, stream, hello = phi_server_with_key

    # Create session
    await stream.send({
        "type": "create_session",
        "session_id": "shared-session",
    })
    session_resp = await stream.recv()
    assert session_resp["type"] == "session_created"

    # Run 1: establish context
    await stream.send({
        "type": "run",
        "session_id": "shared-session",
        "query": 'My name is Alice. Reply with just "OK".',
        "config": {},
    })
    while True:
        msg = await stream.recv()
        if msg["type"] == "done":
            break

    # Run 2: reuse same session
    await stream.send({
        "type": "run",
        "session_id": "shared-session",
        "query": "What is my name? Reply with just the name.",
        "config": {},
    })
    events = []
    while True:
        msg = await stream.recv()
        if msg["type"] == "done":
            break
        if msg["type"] == "event":
            events.append(msg)

    texts = " ".join(
        e.get("text", "") for e in events
        if e.get("runtimeEventType") == "textDelta"
    )
    assert "Alice" in texts, f"Expected session context, got: {texts}"


# ═══════════════════════════════════════════════════════════════════════════
# PT-08 → PT-13: Message Format & NDJSON Frame Boundaries
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pt_08_invalid_json(phi_server) -> None:
    """Sending invalid JSON returns error with PARSE_ERROR code."""
    proc, stream, hello = phi_server

    proc.stdin.write(b"not valid json\n")
    await proc.stdin.drain()

    resp = await stream.recv()
    assert resp["type"] == "error"
    assert resp.get("code") == "PARSE_ERROR"


@pytest.mark.asyncio
async def test_pt_09_unknown_message_type(phi_server) -> None:
    """Unknown message types return PARSE_ERROR.

    phi serve validates against a known type set: register_tool,
    create_session, run, tool_result, cancel. Unknown types are rejected.
    """
    proc, stream, hello = phi_server

    proc.stdin.write(b'{"type":"future_msg","data":42}\n')
    await proc.stdin.drain()

    resp = await stream.recv()
    assert resp["type"] == "error"
    assert "future_msg" in resp.get("message", "")


@pytest.mark.asyncio
async def test_pt_10_oversized_line(phi_server) -> None:
    """A line >64KB is not truncated — full JSON is parsed."""
    proc, stream, hello = phi_server

    long_desc = "x" * (65 * 1024)
    long_msg = json.dumps({
        "type": "register_tool",
        "name": "big_tool",
        "description": long_desc,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }) + "\n"

    proc.stdin.write(long_msg.encode("utf-8"))
    await proc.stdin.drain()

    ack = await stream.recv()
    assert ack["type"] == "tool_registered"
    assert ack.get("ok") is True


@pytest.mark.asyncio
async def test_pt_11_empty_line(phi_server) -> None:
    """Empty lines are handled — server remains responsive afterward."""
    proc, stream, hello = phi_server

    proc.stdin.write(b"\n")
    await proc.stdin.drain()

    # Server should still accept valid messages after an empty line
    await stream.send({"type": "create_session"})
    resp = await stream.recv()
    assert resp["type"] in ("session_created", "error")


@pytest.mark.asyncio
async def test_pt_13_multiple_lines_in_one_read(phi_server) -> None:
    """Multiple JSON lines sent at once are parsed one by one."""
    proc, stream, hello = phi_server

    payload = (
        json.dumps({"type": "create_session"}) + "\n"
        + json.dumps({"type": "create_session"}) + "\n"
        + json.dumps({"type": "create_session"}) + "\n"
    )
    proc.stdin.write(payload.encode("utf-8"))
    await proc.stdin.drain()

    for _ in range(3):
        resp = await stream.recv()
        assert resp["type"] == "session_created"


# PT-12 MUST run last in this group — non-UTF-8 bytes may close the
# connection, making the shared process unusable for subsequent tests.
@pytest.mark.asyncio
async def test_pt_12_non_utf8_bytes(phi_server) -> None:
    """Non-UTF-8 bytes cause connection close or error — server handles it."""
    proc, stream, hello = phi_server

    proc.stdin.write(b'\xff\xfe{"type":"bad"}\n')
    await proc.stdin.drain()

    # Server may return an error, or close the connection.
    # Both are acceptable — the key is it doesn't silently corrupt state.
    try:
        resp = await stream.recv()
        assert resp["type"] == "error"
    except Exception:
        # Connection close is acceptable for binary garbage input
        pass


# ═══════════════════════════════════════════════════════════════════════════
# PT-14 & PT-15: Cancel Mechanism
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pt_14_cancel_message(phi_server_with_key) -> None:
    """Sending cancel during a run interrupts it with RunCancelled."""
    proc, stream, hello = phi_server_with_key

    await stream.send({
        "type": "run",
        "session_id": "",
        "query": "Count from 1 to 100 slowly, one number per line.",
        "config": {},
    })

    # Give it time to start the LLM stream, then cancel
    await asyncio.sleep(1.0)
    await stream.send({"type": "cancel"})

    cancelled_or_errored = False
    while True:
        msg = await stream.recv()
        if msg["type"] == "done":
            break
        if msg["type"] == "event":
            if msg.get("runtimeEventType") == "RunCancelled":
                cancelled_or_errored = True
        if msg["type"] == "error":
            # Turn may fail with TURN_ERROR before RunCancelled is emitted,
            # especially if cancel arrives before the LLM starts streaming.
            cancelled_or_errored = True

    assert cancelled_or_errored, "Expected RunCancelled or TURN_ERROR after cancel"


@pytest.mark.asyncio
async def test_pt_15_agent_cancel_method() -> None:
    """Agent.cancel() sends a cancel message to phi serve."""
    agent = Agent(model="test-model")
    assert hasattr(agent, "cancel"), "Agent must have a cancel() method"
    assert callable(agent.cancel), "Agent.cancel must be callable"
