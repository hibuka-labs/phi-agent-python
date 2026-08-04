"""Shared fixtures for protocol integration tests.

Spawns a real phi serve subprocess (workspace build) for each test module.

- ``phi_server`` — no API key (pure protocol tests)
- ``phi_server_with_key`` — loads .env, includes LLM_API_KEY (for tests
  that need the LLM, e.g. cancel, session reuse)
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from phi_agent.protocol import ProtocolStream

# Path to workspace phi binary
_WORKSPACE_PHI = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "phi-agent" / "target" / "debug" / "phi"
)


def _get_phi_binary() -> str:
    if _WORKSPACE_PHI.exists():
        return str(_WORKSPACE_PHI)
    env = os.environ.get("PHI_PATH")
    if env:
        return env
    import shutil
    which = shutil.which("phi")
    if which:
        return which
    raise FileNotFoundError("phi binary not found")


def _load_dotenv() -> dict[str, str]:
    """Load .env from project root."""
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_file.exists():
        return {}
    vars_: dict[str, str] = {}
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            vars_[key] = val
    return vars_


async def _start_phi(env: dict[str, str]):
    """Start phi serve with given env, return (proc, stream, hello, drain_task)."""
    phi = _get_phi_binary()
    proc = await asyncio.create_subprocess_exec(
        phi, "serve",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stream = ProtocolStream(
        reader=proc.stdout,  # type: ignore[arg-type]
        writer=proc.stdin,  # type: ignore[arg-type]
    )

    async def _drain() -> None:
        if proc.stderr is None:
            return
        while True:
            line = await proc.stderr.readline()
            if not line:
                break

    drain_task = asyncio.create_task(_drain())
    hello = await stream.recv()
    assert hello.get("type") == "hello"
    return proc, stream, hello, drain_task


async def _stop_phi(proc, stream, drain_task) -> None:
    """Stop phi serve and clean up."""
    drain_task.cancel()
    try:
        await drain_task
    except asyncio.CancelledError:
        pass
    await stream.close()
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5)
    except (ProcessLookupError, TimeoutError, asyncio.TimeoutError):
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass


@pytest.fixture(scope="module")
async def phi_server() -> AsyncIterator[
    tuple[asyncio.subprocess.Process, ProtocolStream, dict]
]:
    """Start phi serve WITHOUT API key — protocol-only tests."""
    env = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"}
    proc, stream, hello, drain_task = await _start_phi(env)
    yield proc, stream, hello
    await _stop_phi(proc, stream, drain_task)


@pytest.fixture(scope="module")
async def phi_server_with_key() -> AsyncIterator[
    tuple[asyncio.subprocess.Process, ProtocolStream, dict]
]:
    """Start phi serve WITH API key from .env — for tests that need LLM."""
    dotenv = _load_dotenv()
    env = {**os.environ, **dotenv}
    proc, stream, hello, drain_task = await _start_phi(env)
    yield proc, stream, hello
    await _stop_phi(proc, stream, drain_task)
