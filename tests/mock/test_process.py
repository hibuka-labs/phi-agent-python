"""MS-13 → MS-17: PhiProcess lifecycle tests.

Tests PhiProcess start, handshake, protocol version checking, close
cleanup, and stderr draining. Uses mocked asyncio.create_subprocess_exec.
No phi binary required.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from phi_agent.process import PhiProcess
from phi_agent.protocol import ProtocolError


# ── Helpers ───────────────────────────────────────────────────────────────


async def _make_mock_subprocess(
    stdout_lines: list[dict] | None = None,
    stderr_lines: list[str] | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    """Create a mock ``asyncio.subprocess.Process`` with controlled I/O.

    Returns ``(mock_proc, mock_stdin_writer)``.
    """
    mock_proc = AsyncMock(spec=asyncio.subprocess.Process)

    # Stdout reader: pre-feed NDJSON lines, then EOF
    stdout_reader = asyncio.StreamReader()
    if stdout_lines:
        for line in stdout_lines:
            payload = json.dumps(line, ensure_ascii=False) + "\n"
            stdout_reader.feed_data(payload.encode("utf-8"))
        stdout_reader.feed_eof()
    mock_proc.stdout = stdout_reader

    # Stderr reader
    stderr_reader = asyncio.StreamReader()
    if stderr_lines:
        for text in stderr_lines:
            stderr_reader.feed_data((text + "\n").encode("utf-8"))
        stderr_reader.feed_eof()
    else:
        stderr_reader.feed_eof()
    mock_proc.stderr = stderr_reader

    # Stdin writer (AsyncMock tracks close/wait_closed/drain calls)
    mock_writer = AsyncMock(spec=asyncio.StreamWriter)
    mock_proc.stdin = mock_writer

    mock_proc.returncode = None
    mock_proc.pid = 12345
    # terminate()/kill() are synchronous on asyncio.subprocess.Process
    mock_proc.terminate = MagicMock()
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    return mock_proc, mock_writer


# ── MS-13: 正常启动 + 握手 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_13_normal_startup_and_handshake() -> None:
    """PhiProcess.start + handshake correctly parse the hello message."""
    mock_proc, _ = await _make_mock_subprocess(
        stdout_lines=[
            {"type": "hello", "protocol_version": 1, "server_version": "0.2.0"},
        ],
    )

    with patch("phi_agent.process.asyncio.create_subprocess_exec", return_value=mock_proc):
        proc = PhiProcess(phi_path="/fake/phi")
        await proc.start()
        hello = await proc.handshake()

    assert proc.protocol_version == 1
    assert hello["type"] == "hello"
    assert hello["server_version"] == "0.2.0"


# ── MS-14: 协议版本不兼容 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_14_protocol_version_incompatible() -> None:
    """ProtocolError raised when the server reports an unsupported version."""
    mock_proc, _ = await _make_mock_subprocess(
        stdout_lines=[
            {"type": "hello", "protocol_version": 99, "server_version": "0.1.0"},
        ],
    )

    with patch("phi_agent.process.asyncio.create_subprocess_exec", return_value=mock_proc):
        proc = PhiProcess(phi_path="/fake/phi")
        await proc.start()
        with pytest.raises(ProtocolError, match="unsupported protocol version 99"):
            await proc.handshake()


# ── MS-15: close() 清理 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_15_close_cleanup() -> None:
    """close() terminates the subprocess and closes stdin/stdout."""
    mock_proc, mock_writer = await _make_mock_subprocess(
        stdout_lines=[
            {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        ],
    )

    with patch("phi_agent.process.asyncio.create_subprocess_exec", return_value=mock_proc):
        proc = PhiProcess(phi_path="/fake/phi")
        await proc.start()
        await proc.handshake()
        await proc.close()

    # Subprocess was terminated
    mock_proc.terminate.assert_called_once()
    # Writer was closed
    mock_writer.close.assert_called_once()
    mock_writer.wait_closed.assert_called_once()


# ── MS-16: 未 close() 退出 (context manager) ─────────────────────────────


@pytest.mark.asyncio
async def test_ms_16_context_manager_cleanup() -> None:
    """The async context manager (__aenter__/__aexit__) properly cleans up."""
    mock_proc, mock_writer = await _make_mock_subprocess(
        stdout_lines=[
            {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        ],
    )

    with patch("phi_agent.process.asyncio.create_subprocess_exec", return_value=mock_proc):
        async with PhiProcess(phi_path="/fake/phi") as proc:
            assert proc.protocol_version == 1

        # After __aexit__, the process should be cleaned up
        mock_proc.terminate.assert_called_once()
        mock_writer.close.assert_called_once()


# ── MS-17: stderr 排水 ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ms_17_stderr_drain_does_not_block() -> None:
    """Background stderr draining reads all lines without blocking."""
    stderr_texts = [
        "WARN  phi::serve starting up",
        "INFO  phi::serve listening",
        "DEBUG phi::serve request processed",
    ]

    mock_proc, _ = await _make_mock_subprocess(
        stdout_lines=[
            {"type": "hello", "protocol_version": 1, "server_version": "0.1.0"},
        ],
        stderr_lines=stderr_texts,
    )

    with patch("phi_agent.process.asyncio.create_subprocess_exec", return_value=mock_proc):
        proc = PhiProcess(phi_path="/fake/phi")
        await proc.start()
        await proc.handshake()

        # Give the background stderr drain task time to consume all lines
        await asyncio.sleep(0.05)

        # close() cancels the stderr drain task — should not hang
        await proc.close()

    # If we got here without hanging, stderr was properly drained
    mock_proc.terminate.assert_called_once()
