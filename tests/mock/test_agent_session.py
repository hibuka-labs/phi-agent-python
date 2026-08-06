"""MS-15 → MS-17: Agent.create_session() tests using mock streams."""

from __future__ import annotations

import pytest

from tests.mock.conftest import MockStream, patch_agent


# ── MS-15: create_session 创建会话 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_15_create_session_with_external_id(mock_stream: MockStream) -> None:
    """Agent.create_session() with explicit session_id."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.2.7"},
        {
            "type": "session_created",
            "session_id": "my-session",
            "internal_id": 42,
        },
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        result = await agent.create_session(session_id="my-session")

    assert result["type"] == "session_created"
    assert result["session_id"] == "my-session"
    assert result["internal_id"] == 42

    # Verify the correct message was sent with session_id
    sent = mock_stream.sent_messages
    assert sent[0]["type"] == "create_session"
    assert sent[0]["session_id"] == "my-session"


# ── MS-16: create_session 不传 session_id ─────────────────────────────────


@pytest.mark.asyncio
async def test_ms_16_create_session_without_id(mock_stream: MockStream) -> None:
    """Agent.create_session() without explicit session_id (server assigns)."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.2.7"},
        {
            "type": "session_created",
            "session_id": None,
            "internal_id": 1,
        },
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        result = await agent.create_session()

    assert result["type"] == "session_created"
    assert result["internal_id"] == 1

    sent = mock_stream.sent_messages
    assert sent[0]["type"] == "create_session"
    assert sent[0]["session_id"] is None


# ── MS-17: create_session 收到非 session_created 响应 ───────────────────


@pytest.mark.asyncio
async def test_ms_17_create_session_unexpected_response(mock_stream: MockStream) -> None:
    """Agent.create_session() raises RuntimeError on unexpected response."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.2.7"},
        {"type": "error", "code": "SESSION_ERROR", "message": "limit reached"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        with pytest.raises(RuntimeError, match="expected session_created"):
            await agent.create_session()
