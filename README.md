# phi-agent — Python SDK

Build AI agents with Python. Write tools in Python, run on a Rust-powered agent runtime.

## Quick Start

```python
import asyncio
from phi_agent import Agent, tool

@tool
async def get_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def main():
    agent = Agent(model="gpt-4o")
    agent.register(get_time)

    async for event in agent.run("What time is it?"):
        match event.type:
            case "thought_delta": print(f"💭 {event.text}", end="", flush=True)
            case "text_delta":    print(event.text, end="", flush=True)
            case "tool_call_started":
                print(f"\n🔧 {event.tool_name}...")
            case "tool_call_finished":
                print(f"  → {event.summary}")

asyncio.run(main())
```

## How It Works

The Python SDK communicates with the `phi` Rust binary via stdio NDJSON protocol.
The Rust runtime handles the agent loop, session management, LLM calls, and event
streaming — you only write tools in Python.

```
Python process                    Rust process
┌──────────────┐    stdio     ┌──────────────────┐
│ phi_agent     │◄─ NDJSON ─►│ phi serve         │
│ (your tools)  │             │ (agent runtime)   │
└──────────────┘             └──────────────────┘
```

- Protocol specification: [phi-agent repo](../phi-agent/docs/protocol.md)

## License

MIT
