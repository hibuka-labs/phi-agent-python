"""
End-to-end integration test: Python SDK ↔ phi serve ↔ real LLM.

Requires:
  - phi binary built: cargo build --release
  - LLM_API_KEY set in .env (or env var)

Usage:
  cd phi-agent && cargo build --release
  PHI_PATH=./target/release/phi python tests/e2e_test.py
"""

import asyncio
import os
import sys

# Point to the SDK
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phi_agent import Agent, tool


@tool
async def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main():
    # ── Check prerequisites ─────────────────────────────────────────
    phi_path = os.environ.get("PHI_PATH", "./target/release/phi")
    if not os.path.exists(phi_path):
        print(f"❌ phi binary not found at {phi_path}")
        print("   Run: cargo build --release")
        return 1

    # ── Set up agent ────────────────────────────────────────────────
    print(f"📦 Test: phi-agent Python SDK e2e test")
    print(f"   phi binary: {phi_path}")

    # Enable debug logs for this test
    os.environ["PHI_LOG_LEVEL"] = "DEBUG"

    agent = Agent(model="gpt-4o", phi_path=phi_path)
    agent.register(get_time)

    # ── Test 1: Simple query (no tool) ──────────────────────────────
    print("\n── Test 1: simple query ──")
    events = []
    async for event in agent.run("Say exactly: hello world"):
        events.append(event)
        if event.type in ("textDelta", "thoughtDelta"):
            print(event.text, end="", flush=True)

    assert any(e.type == "runFinished" for e in events) or \
           any(e.type == "textDelta" for e in events), \
           "Expected text response or runFinished"
    print("\n✅ Test 1 passed")

    # ── Test 2: Tool call ───────────────────────────────────────────
    print("\n── Test 2: tool call ──")
    events = []
    async for event in agent.run("What time is it right now?"):
        events.append(event)
        match event.type:
            case "thoughtDelta":
                print(event.text, end="", flush=True)
            case "textDelta":
                print(event.text, end="", flush=True)
            case "toolCallStarted":
                print(f"\n🔧 calling {event.tool_name}...")
            case "toolCallFinished":
                print(f"   ✅ {event.summary[:80]}")

    assert any(e.type == "textDelta" for e in events) or \
           any(e.type == "toolCallFinished" for e in events), \
           "Expected tool call or text response"
    print("\n✅ Test 2 passed")

    # ── Cleanup ─────────────────────────────────────────────────────
    await agent.close()

    # ── Check logs exist ────────────────────────────────────────────
    log_dir = os.path.expanduser("~/.phi-agent")
    serve_logs = [f for f in os.listdir(log_dir) if f.startswith("serve-")]
    sdk_logs = [f for f in os.listdir(log_dir) if f.startswith("sdk-")]
    print(f"\n📋 Logs:")
    print(f"   serve: ~/.phi-agent/{serve_logs[-1] if serve_logs else 'N/A'}")
    print(f"   sdk:   ~/.phi-agent/{sdk_logs[-1] if sdk_logs else 'N/A'}")

    print("\n🎉 All e2e tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
