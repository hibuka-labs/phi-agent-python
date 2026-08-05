"""
phi-agent Python SDK — 5-minute demo

Prerequisites:
  1. pip install phi-agent
  2. phi binary on PATH, or set PHI_PATH
  3. .env with LLM_API_KEY, LLM_MODEL, LLM_BASE_URL

Usage:  python demo/demo.py
"""

import asyncio
import os
from pathlib import Path

from phi_agent import Agent, tool


# ── Define your tools ────────────────────────────────────────────────────

@tool
async def get_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
async def search(query: str) -> str:
    """Search for information."""
    return f'Results for "{query}": phi-agent is a Python SDK powered by Rust runtime.'


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    phi_path = os.environ.get("PHI_PATH")
    if not phi_path:
        bundled = Path(__file__).resolve().parent.parent / "bin" / "phi"
        if bundled.exists():
            phi_path = str(bundled)

    agent = Agent(
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )
    agent.register(get_time)
    agent.register(search)

    try:
        # Round 1: tool call
        print("═" * 50)
        print("Round 1: 调用 get_time 工具")
        print("═" * 50)
        async for event in agent.run("What time is it now? Use the get_time tool."):
            match event.type:
                case "thoughtDelta":
                    print(f"[思考] {event.text}", end="", flush=True)
                case "textDelta":
                    print(event.text, end="", flush=True)
                case "toolCallStarted":
                    print(f"\n🔧 {event.tool_name}")
                case "toolCallFinished":
                    if event.summary:
                        print(f"   → {event.summary[:120]}")
        print("\n")

        # Round 2: another tool call
        print("═" * 50)
        print("Round 2: 调用 search 工具")
        print("═" * 50)
        async for event in agent.run("Search for 'phi-agent' and tell me what it is."):
            match event.type:
                case "thoughtDelta":
                    print(f"[思考] {event.text}", end="", flush=True)
                case "textDelta":
                    print(event.text, end="", flush=True)
                case "toolCallStarted":
                    print(f"\n🔧 {event.tool_name}")
                case "toolCallFinished":
                    if event.summary:
                        print(f"   → {event.summary[:120]}")
        print("\n🎉 Done!")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
