"""
phi-agent Library Integration — one-shot agent call

Usage:  python demo/lib.py

Shows how to use phi-agent as a library in your Python app:
define tools → register → run a single query → get results.
"""

import asyncio
import os
from pathlib import Path

from phi_agent import Agent, tool


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_dotenv():
    """Load .env from project root into os.environ (best-effort)."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# ── Step 1: Define your tools ────────────────────────────────────────────

@tool
async def get_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
async def search(query: str) -> str:
    """Search for information (mock)."""
    return f'Results for "{query}": phi-agent is a Python SDK for building AI agents.'


# ── Step 2: Create agent and register tools ──────────────────────────────

async def main():
    agent = Agent(
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
    )
    agent.register(get_time)
    agent.register(search)

    # ── Step 3: Run ──────────────────────────────────────────────────
    try:
        thinking = False
        async for event in agent.run("What time is it? Use get_time."):
            match event.type:
                case "thoughtDelta":
                    thinking = True
                    print(f"\033[90m{event.text}\033[0m", end="", flush=True)
                case "textDelta":
                    if thinking:
                        print()
                        thinking = False
                    print(event.text, end="", flush=True)
                case "toolCallStarted":
                    if thinking:
                        print()
                        thinking = False
                    print(f"\n🔧 {event.tool_name}")
                case "toolCallFinished":
                    if event.summary:
                        print(f"   → {event.summary[:120]}")
        print()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
