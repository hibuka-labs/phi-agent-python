"""
phi-agent REPL — interactive agent session

Usage:  python demo/repl.py
Type /exit to quit, /tools to list tools, /reset to start new session.
"""

import asyncio
import json
import os
from pathlib import Path

# Fix Chinese backspace on macOS (libedit bug)
try:
    import gnureadline  # noqa: F401
except ImportError:
    pass

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


# ── Tools ────────────────────────────────────────────────────────────────

@tool
async def get_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Event rendering ──────────────────────────────────────────────────────

def make_renderer():
    """Create an event renderer with internal state for thinking/text separation."""
    thinking = False

    def render(event) -> None:
        nonlocal thinking
        match event.type:
            case "thoughtDelta":
                thinking = True
                print(f"\033[90m{event.text}\033[0m", end="", flush=True)
            case "textDelta":
                if thinking:
                    print()  # end thinking line
                    thinking = False
                print(event.text, end="", flush=True)
            case "toolCallStarted":
                if thinking:
                    print()
                    thinking = False
                args = event.data.get("args_json") or event.args_json
                args_str = ""
                if args and args != "{}":
                    try:
                        args_str = json.dumps(json.loads(args), ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        args_str = args
                print(f"\n🔧 {event.tool_name} {args_str}")
            case "toolCallFinished":
                summary = event.summary
                if summary:
                    print(f"   → {summary[:200]}")
            case _:
                pass

    return render


# ── REPL ─────────────────────────────────────────────────────────────────

def print_banner(model: str):
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  phi-agent Python SDK — REPL                 ║")
    print(f"║  Model:  {model:<35}║")
    print("║  Commands: /exit | /tools | /reset          ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print("Try: 现在几点了？")
    print()


async def main():
    _load_dotenv()

    model = os.environ.get("LLM_MODEL", "gpt-4o")

    def make_agent():
        a = Agent(
            model=model,
            api_key=os.environ.get("LLM_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL"),
        )
        a.register(get_time)
        return a

    agent = make_agent()
    print_banner(model)

    try:
        while True:
            try:
                user_input = input("\nphi> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input == "/exit":
                break
            if user_input == "/reset":
                await agent.close()
                agent = make_agent()
                print("✅ New session created\n")
                continue
            if user_input == "/tools":
                tools = list(agent._tools.keys())
                if tools:
                    print(f"\n  Registered tools ({len(tools)}):")
                    for name in tools:
                        t = agent._tools[name]
                        print(f"  \033[1m{name}\033[0m")
                        print(f"    {t.description}")
                else:
                    print("\n  (no tools registered)")
                print()
                continue

            render = make_renderer()
            try:
                async for event in agent.run(user_input):
                    render(event)
            except (KeyboardInterrupt, asyncio.CancelledError):
                try:
                    await agent.cancel()
                except Exception:
                    pass
                print("\n⏹ Cancelled")
            except Exception as e:
                print(f"\n❌ Error: {e}")
            print()
    finally:
        try:
            await agent.close()
        except BaseException:
            pass  # cleanup interrupted — harmless


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
