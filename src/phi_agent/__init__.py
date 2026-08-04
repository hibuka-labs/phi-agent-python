"""
phi-agent Python SDK

Build AI agents with Python.  Write tools in Python, run on a Rust-powered
agent runtime via the stdio NDJSON protocol.

Quick start::

    import asyncio
    from phi_agent import Agent, tool

    @tool
    async def search(query: str) -> str:
        '''Search the web.'''
        ...

    async def main():
        agent = Agent(model="gpt-4o")
        agent.register(search)

        async for event in agent.run("Find the latest news"):
            print(event)

    asyncio.run(main())
"""

from .agent import Agent
from .events import Event
from .tool import ToolOutput, tool

__all__ = ["Agent", "Event", "ToolOutput", "tool"]
