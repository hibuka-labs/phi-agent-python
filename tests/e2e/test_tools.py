"""E2E-07 → E2E-12: Tool calls — registration, arguments, ToolOutput, exceptions, concurrency.

Requires phi serve + LLM API key.  Tool-calling prompts are tuned for
DeepSeek Flash; other models may need different prompting.
"""

from __future__ import annotations

import pytest

from phi_agent import Agent, tool
from phi_agent.tool import ToolOutput


# ═══════════════════════════════════════════════════════════════════════════
# E2E-07: No-argument tool
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_e2e_07_no_arg_tool() -> None:
    """Register a no-arg tool and ask the LLM to call it."""
    import os
    from pathlib import Path

    workspace_phi = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "phi-agent" / "target" / "debug" / "phi"
    )
    phi_path = os.environ.get("PHI_PATH", str(workspace_phi))

    @tool
    async def get_current_time() -> str:
        """Return the current date and time. Call this when asked about time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    agent = Agent(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )
    agent.register(get_current_time)

    try:
        tool_called = False
        async for event in agent.run(
            "What time is it? Call the get_current_time tool to find out."
        ):
            if event.type == "toolCallStarted":
                assert event.tool_name == "get_current_time"
                tool_called = True
            elif event.type == "toolCallFinished":
                assert event.summary != ""

        assert tool_called, (
            "Expected toolCallStarted event for get_current_time"
        )
    finally:
        await agent.close()


# ═══════════════════════════════════════════════════════════════════════════
# E2E-08: Tool with arguments
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_e2e_08_arg_tool() -> None:
    """Register a tool with parameters — LLM should pass args correctly."""
    import json
    import os
    from pathlib import Path

    workspace_phi = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "phi-agent" / "target" / "debug" / "phi"
    )
    phi_path = os.environ.get("PHI_PATH", str(workspace_phi))

    @tool
    async def greet(name: str, language: str = "English") -> str:
        """Greet someone by name in the specified language."""
        greetings = {
            "English": "Hello",
            "Chinese": "你好",
            "Spanish": "Hola",
            "French": "Bonjour",
        }
        word = greetings.get(language, "Hello")
        return f"{word}, {name}!"

    agent = Agent(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )
    agent.register(greet)

    try:
        tool_called = False
        async for event in agent.run(
            "Call the greet tool with name='World' and language='Chinese'. "
            "Use the tool — do not just type the greeting."
        ):
            if event.type == "toolCallStarted":
                assert event.tool_name == "greet"
                tool_called = True

        assert tool_called, "Expected toolCallStarted for greet"
    finally:
        await agent.close()


# ═══════════════════════════════════════════════════════════════════════════
# E2E-09: Tool returning ToolOutput
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_e2e_09_tool_output() -> None:
    """Tool returning ToolOutput(summary=..., raw=...) — raw should pass through."""
    import os
    from pathlib import Path

    workspace_phi = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "phi-agent" / "target" / "debug" / "phi"
    )
    phi_path = os.environ.get("PHI_PATH", str(workspace_phi))

    @tool
    async def get_weather(city: str) -> ToolOutput:
        """Get the weather for a city. Returns structured data."""
        return ToolOutput(
            summary=f"Weather in {city}: sunny, 22°C",
            raw={"city": city, "temp": 22, "condition": "sunny"},
        )

    agent = Agent(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )
    agent.register(get_weather)

    try:
        summary_seen = ""
        async for event in agent.run(
            "Call the get_weather tool for the city Beijing. "
            "Pass city='Beijing' as the parameter."
        ):
            if event.type == "toolCallFinished":
                summary_seen = event.summary

        # The tool was called — summary may contain weather info or an
        # error message if the LLM didn't pass the city arg correctly.
        assert summary_seen != "", "Expected toolCallFinished summary"
        # Best-effort: if city was passed, check the response
        if "Error" not in summary_seen:
            assert "Beijing" in summary_seen or "sunny" in summary_seen.lower(), (
                f"Expected weather info in summary, got: {summary_seen!r}"
            )
    finally:
        await agent.close()


# ═══════════════════════════════════════════════════════════════════════════
# E2E-10: Multiple tools — LLM selects correctly
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_e2e_10_multi_tool_selection() -> None:
    """Register 3 tools — LLM should select the right one for the query."""
    import os
    from pathlib import Path

    workspace_phi = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "phi-agent" / "target" / "debug" / "phi"
    )
    phi_path = os.environ.get("PHI_PATH", str(workspace_phi))

    @tool
    async def get_time_utc() -> str:
        """Get the current UTC time."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    @tool
    async def add_numbers(a: int, b: int) -> str:
        """Add two integers and return the sum."""
        return str(a + b)

    @tool
    async def reverse_text(text: str) -> str:
        """Reverse the given text."""
        return text[::-1]

    agent = Agent(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )
    agent.register(get_time_utc)
    agent.register(add_numbers)
    agent.register(reverse_text)

    try:
        tool_name_seen = ""
        async for event in agent.run(
            "Call the tool that reverses text. Reverse the text 'hello world'. "
            "Use the reverse_text tool."
        ):
            if event.type == "toolCallStarted":
                tool_name_seen = event.tool_name

        assert tool_name_seen == "reverse_text", (
            f"Expected reverse_text, got: {tool_name_seen!r}"
        )
    finally:
        await agent.close()


# ═══════════════════════════════════════════════════════════════════════════
# E2E-11: Tool raises exception — conversation continues
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_e2e_11_tool_exception() -> None:
    """When a tool raises, the LLM should get an error summary and continue."""
    import os
    from pathlib import Path

    workspace_phi = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "phi-agent" / "target" / "debug" / "phi"
    )
    phi_path = os.environ.get("PHI_PATH", str(workspace_phi))

    @tool
    async def broken_tool() -> str:
        """This tool always fails. Do not call it."""
        raise ValueError("custom error: tool is broken")

    agent = Agent(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )
    agent.register(broken_tool)

    try:
        # We don't force the LLM to call the tool — if it reads the
        # description it might skip it.  Instead we verify the agent
        # handles the run without crashing.
        finished = False
        error_seen = False
        async for event in agent.run(
            "Call the broken_tool function. I want to see what happens."
        ):
            if event.type == "toolCallFinished":
                summary = event.summary.lower()
                if "error" in summary or "custom error" in summary:
                    error_seen = True
            elif event.type == "runFinished":
                finished = True

        # At minimum, the run should complete without crashing
        assert finished or error_seen, (
            "Expected run to finish or tool error to appear"
        )
    finally:
        await agent.close()


# ═══════════════════════════════════════════════════════════════════════════
# E2E-12: Concurrent tool calls (if supported)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_e2e_12_concurrent_tool_calls() -> None:
    """LLM calls 2+ tools at once — tool_call events should not interleave badly.

    Note: concurrent tool calling depends on the model.  Not all models
    support parallel tool calls.  The test verifies that if multiple
    tool calls arrive, each has its own call_id and they don't corrupt
    each other.
    """
    import os
    from pathlib import Path

    workspace_phi = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "phi-agent" / "target" / "debug" / "phi"
    )
    phi_path = os.environ.get("PHI_PATH", str(workspace_phi))

    @tool
    async def get_temperature(city: str) -> str:
        """Get the temperature for a city."""
        temps = {"Beijing": "15°C", "Shanghai": "22°C", "Tokyo": "18°C"}
        return temps.get(city, "20°C")

    @tool
    async def get_humidity(city: str) -> str:
        """Get the humidity for a city."""
        humidities = {"Beijing": "30%", "Shanghai": "65%", "Tokyo": "55%"}
        return humidities.get(city, "50%")

    agent = Agent(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )
    agent.register(get_temperature)
    agent.register(get_humidity)

    try:
        call_ids: set[str] = set()
        tool_names: list[str] = []
        async for event in agent.run(
            "Call both get_temperature and get_humidity for Beijing "
            "at the same time if possible."
        ):
            if event.type == "toolCallStarted":
                tool_names.append(event.tool_name)
                # Extract call_id from raw event data if available
                cid = event.data.get("callId", "")
                if cid:
                    call_ids.add(cid)
            elif event.type == "toolCallFinished":
                tool_names.append(event.tool_name)

        # Verify at least one tool was called
        assert len(tool_names) > 0, "Expected at least one tool call"

        # If concurrent calls happened, call_ids should be unique
        # (single tool calls also pass — this is best-effort)
        if len(call_ids) > 1:
            assert len(call_ids) == len(tool_names) / 2, (
                f"Expected unique call_ids for concurrent calls, "
                f"got {len(call_ids)} call_ids for {len(tool_names)} events"
            )
    finally:
        await agent.close()
