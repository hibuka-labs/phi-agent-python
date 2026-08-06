"""MS-12 → MS-14: Agent.list_tools() tests using mock streams."""

from __future__ import annotations

import pytest

from tests.mock.conftest import MockStream, patch_agent


# ── MS-12: list_tools 返回工具列表 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_12_list_tools_returns_metadata(mock_stream: MockStream) -> None:
    """Agent.list_tools() returns tool metadata from the server."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.2.7"},
        {
            "type": "tools_listed",
            "tools": [
                {
                    "name": "shell",
                    "description": "Run shell commands",
                    "origin": "phi-tools",
                    "version": "1.0.0",
                    "requirements": ["bash"],
                },
                {
                    "name": "search",
                    "description": "Search the web",
                    "origin": "user",
                    "version": "0.1.0",
                    "requirements": [],
                },
            ],
        },
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        tools = await agent.list_tools()

    assert len(tools) == 2
    assert tools[0]["name"] == "shell"
    assert tools[0]["origin"] == "phi-tools"
    assert tools[0]["requirements"] == ["bash"]
    assert tools[1]["name"] == "search"
    assert tools[1]["origin"] == "user"

    # Verify the correct message was sent
    sent = mock_stream.sent_messages
    assert sent[0]["type"] == "list_tools"


# ── MS-13: list_tools 空列表 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_13_list_tools_empty(mock_stream: MockStream) -> None:
    """Agent.list_tools() returns empty list when no tools registered."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.2.7"},
        {"type": "tools_listed", "tools": []},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        tools = await agent.list_tools()

    assert tools == []


# ── MS-14: list_tools 收到非 tools_listed 响应 ──────────────────────────


@pytest.mark.asyncio
async def test_ms_14_list_tools_unexpected_response(mock_stream: MockStream) -> None:
    """Agent.list_tools() raises RuntimeError on unexpected response type."""
    mock_stream.feed(
        {"type": "hello", "protocol_version": 1, "server_version": "0.2.7"},
        {"type": "error", "code": "INTERNAL", "message": "something broke"},
    )

    with patch_agent(mock_stream, model="test-model") as agent:
        with pytest.raises(RuntimeError, match="expected tools_listed"):
            await agent.list_tools()
