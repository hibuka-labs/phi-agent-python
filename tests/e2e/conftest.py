"""E2E test fixtures — real phi serve + LLM API key.

Each test module gets its own phi serve process (module scope).  Tests that
need destructive operations (kill, SIGINT) should create their own Agent
instances inline rather than using the shared fixture.

Markers:
    @pytest.mark.e2e    — requires phi serve + LLM key
    @pytest.mark.slow   — may take >60s (deep thinking models)
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from phi_agent import Agent
from phi_agent.tool import tool

# ── Path resolution ──────────────────────────────────────────────────────

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_PHI_BINARY = _WORKSPACE_ROOT / "phi-agent" / "target" / "debug" / "phi"


def _find_phi_binary() -> str:
    """Resolve phi binary: PHI_PATH env → workspace build → PATH."""
    env = os.environ.get("PHI_PATH")
    if env:
        return env
    if _PHI_BINARY.exists():
        return str(_PHI_BINARY)
    import shutil
    which = shutil.which("phi")
    if which:
        return which
    raise FileNotFoundError(
        f"phi binary not found.  Build it or set PHI_PATH.  "
        f"Expected at: {_PHI_BINARY}"
    )


def _load_dotenv() -> dict[str, str]:
    """Load .env from project root, returning key-value pairs."""
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


# ── Common test tools ─────────────────────────────────────────────────────


@tool
async def get_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
async def search(query: str) -> str:
    """Search for information (mock — returns a fixed answer)."""
    return f"Search results for: {query} — no real results (test mock)"


@tool
async def echo(text: str) -> str:
    """Echo back the given text."""
    return text


@tool
async def slow_echo(text: str) -> str:
    """Echo back the text after a delay (for cancel testing)."""
    import asyncio
    await asyncio.sleep(2.0)
    return text


@tool
async def always_fail() -> str:
    """A tool that always raises an error."""
    raise ValueError("This tool always fails — by design")


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
async def e2e_agent() -> AsyncIterator[Agent]:
    """Module-scoped Agent connected to a real phi serve + LLM.

    Starts phi serve once per test module, registers no tools by default.
    Tests can call ``agent.register(tool)`` before ``agent.run()`` as needed.

    The agent is closed automatically after all tests in the module finish.
    """
    dotenv = _load_dotenv()
    env = {**os.environ, **dotenv}

    phi_path = _find_phi_binary()

    agent = Agent(
        model=env.get("LLM_MODEL", "gpt-4o"),
        api_key=env.get("LLM_API_KEY"),
        base_url=env.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )

    yield agent

    # Cleanup
    try:
        await asyncio_sleep(0.5)  # let pending writes flush
        await agent.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
async def e2e_agent_with_tools() -> AsyncIterator[Agent]:
    """Module-scoped Agent with common tools pre-registered.

    Includes: get_time, search, echo.
    """
    dotenv = _load_dotenv()
    env = {**os.environ, **dotenv}

    agent = Agent(
        model=env.get("LLM_MODEL", "gpt-4o"),
        api_key=env.get("LLM_API_KEY"),
        base_url=env.get("LLM_BASE_URL"),
        phi_path=_find_phi_binary(),
    )
    agent.register(get_time)
    agent.register(search)
    agent.register(echo)

    yield agent

    try:
        await asyncio_sleep(0.5)
        await agent.close()
    except Exception:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────

# Import here to avoid name collision with asyncio.sleep
import asyncio as _asyncio


async def asyncio_sleep(seconds: float) -> None:
    """Short alias for tests."""
    await _asyncio.sleep(seconds)


async def collect_events(agent: Agent, query: str, **kwargs) -> list:
    """Run a query and collect all events into a list."""
    events = []
    async for event in agent.run(query, **kwargs):
        events.append(event)
    return events


async def run_and_get_text(agent: Agent, query: str, **kwargs) -> str:
    """Run a query and return the concatenated textDelta text."""
    parts: list[str] = []
    async for event in agent.run(query, **kwargs):
        if event.type == "textDelta":
            parts.append(event.text)
    return "".join(parts)
