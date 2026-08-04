"""Mock test infrastructure for phi-agent — no phi binary needed.

Provides MockStream (test double for ProtocolStream) and MockPhiProcess
(test double for PhiProcess), plus shared pytest fixtures.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from phi_agent.agent import Agent
from phi_agent.process import PhiProcess
from phi_agent.protocol import ProtocolError


# ── Mock Stream ──────────────────────────────────────────────────────────


class MockStream:
    """Test double for :class:`phi_agent.protocol.ProtocolStream`.

    Uses ``asyncio.Queue`` for ``recv()`` and a list for ``send()``.
    Supports injecting exceptions to simulate connection failures.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any] | BaseException] = asyncio.Queue()
        self.sent_messages: list[dict[str, Any]] = []
        self._closed = False

    async def recv(self) -> dict[str, Any]:
        """Return the next pre-loaded message, or raise an injected exception."""
        if self._closed:
            raise ProtocolError("phi process closed stdout unexpectedly")
        msg = await self._queue.get()
        if isinstance(msg, BaseException):
            raise msg
        # None is the sentinel for EOF
        if msg is None:
            raise ProtocolError("phi process closed stdout unexpectedly")
        return msg

    async def send(self, msg: dict[str, Any]) -> None:
        """Record the sent message."""
        if self._closed:
            raise ProtocolError("stream closed")
        self.sent_messages.append(msg)

    async def close(self) -> None:
        """Mark the stream as closed."""
        self._closed = True

    def feed(self, *messages: dict[str, Any] | BaseException | None) -> None:
        """Queue messages (or exceptions) to be returned by ``recv()``.

        Pass ``None`` to simulate EOF (connection close).
        Pass an exception instance to simulate a protocol error on recv.
        """
        for msg in messages:
            self._queue.put_nowait(msg)


# ── Mock PhiProcess ──────────────────────────────────────────────────────


class MockPhiProcess:
    """Test double for :class:`phi_agent.process.PhiProcess`.

    Uses a :class:`MockStream` instead of spawning a real subprocess.
    Replicates the handshake logic from the real PhiProcess.
    """

    def __init__(
        self,
        stream: MockStream,
        phi_path: str | None = None,
    ) -> None:
        self.stream: MockStream = stream
        self._protocol_version: int = 0
        self._server_version: str = ""
        self._closed: bool = False

    @property
    def protocol_version(self) -> int:
        return self._protocol_version

    async def start(self) -> None:
        """No-op — no subprocess to spawn."""

    async def handshake(self) -> dict[str, Any]:
        """Read the hello message and validate protocol version.

        Mirrors :meth:`PhiProcess.handshake`.
        """
        hello = await self.stream.recv()
        if hello.get("type") != "hello":
            raise ProtocolError(f"expected hello, got {hello.get('type')}")
        self._protocol_version = hello.get("protocol_version", 0)
        self._server_version = hello.get("server_version", "unknown")
        if self._protocol_version != 1:
            raise ProtocolError(
                f"unsupported protocol version {self._protocol_version}"
            )
        return hello

    async def close(self) -> None:
        """Mark the process as closed."""
        self._closed = True


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_stream() -> MockStream:
    """A fresh :class:`MockStream` for each test."""
    return MockStream()


@pytest.fixture
def mock_process(mock_stream: MockStream) -> MockPhiProcess:
    """A :class:`MockPhiProcess` wired to the test's mock stream."""
    return MockPhiProcess(mock_stream)


# ── Shared helpers ────────────────────────────────────────────────────────


@contextmanager
def patch_agent(stream: MockStream, **kwargs) -> Iterator[Agent]:
    """Context manager that patches PhiProcess so Agent uses *stream*.

    The patch is active for the entire ``with`` block, so both Agent
    construction and ``run()`` calls see the mock.
    """
    mock_proc = MockPhiProcess(stream)
    with patch("phi_agent.agent.PhiProcess", return_value=mock_proc):
        yield Agent(**kwargs)
