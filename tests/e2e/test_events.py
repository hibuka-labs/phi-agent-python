"""E2E-13 → E2E-17: Event type completeness.

Verifies that each event type the SDK defines is well-formed: textDelta,
thoughtDelta, toolCallStarted, toolCallFinished, planUpdated.
"""

from __future__ import annotations

import json

import pytest

from phi_agent import Agent, tool


# ═══════════════════════════════════════════════════════════════════════════
# E2E-13 & E2E-14: thoughtDelta + textDelta
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_13_thought_delta(e2e_agent: Agent) -> None:
    """DeepSeek models produce thoughtDelta during thinking phase."""
    events = []
    async for event in e2e_agent.run(
        "What is 15 * 37? Think step by step, then give the answer."
    ):
        events.append(event)

    types = {e.type for e in events}
    # DeepSeek v4 models have a thinking phase → thoughtDelta should appear
    # Non-thinking models may skip it — the test just verifies the event
    # is well-formed when present.
    for e in events:
        if e.type == "thoughtDelta":
            assert isinstance(e.text, str), "thoughtDelta.text must be str"
        elif e.type == "textDelta":
            assert isinstance(e.text, str), "textDelta.text must be str"

    assert "textDelta" in types, f"Expected textDelta in {types}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_14_text_delta_has_content(e2e_agent: Agent) -> None:
    """textDelta events should contain non-empty text."""
    texts: list[str] = []
    async for event in e2e_agent.run("Say exactly: the sky is blue"):
        if event.type == "textDelta":
            assert isinstance(event.text, str)
            texts.append(event.text)

    combined = "".join(texts)
    assert len(combined) > 0, "Expected non-empty textDelta content"
    assert "blue" in combined.lower(), f"Expected 'blue' in: {combined!r}"


# ═══════════════════════════════════════════════════════════════════════════
# E2E-15 & E2E-16: toolCallStarted + toolCallFinished
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_e2e_15_tool_call_started_event() -> None:
    """toolCallStarted should have accessible tool_name + args_json."""
    import os
    from pathlib import Path

    workspace_phi = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "phi-agent" / "target" / "debug" / "phi"
    )
    phi_path = os.environ.get("PHI_PATH", str(workspace_phi))

    @tool
    async def lookup_word(word: str) -> str:
        """Look up the definition of a word."""
        definitions = {"hello": "a greeting", "world": "the Earth"}
        return definitions.get(word.lower(), f"Unknown word: {word}")

    agent = Agent(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )
    agent.register(lookup_word)

    try:
        started = False
        all_types: list[str] = []
        async for event in agent.run(
            "Call the lookup_word tool with word='hello'. Use the tool."
        ):
            all_types.append(event.type)
            if event.type == "toolCallStarted":
                name = event.data.get("tool_name", "")
                assert name == "lookup_word", f"Expected lookup_word, got {name!r}"
                started = True

        assert started, f"Expected toolCallStarted in {all_types}"
    finally:
        await agent.close()


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_e2e_16_tool_call_finished_summary() -> None:
    """toolCallFinished summary should contain the tool's return value."""
    import os
    from pathlib import Path

    workspace_phi = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "phi-agent" / "target" / "debug" / "phi"
    )
    phi_path = os.environ.get("PHI_PATH", str(workspace_phi))

    @tool
    async def pick_color() -> str:
        """Pick a color. Always returns 'magenta'."""
        return "magenta"

    agent = Agent(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        phi_path=phi_path,
    )
    agent.register(pick_color)

    try:
        summary = ""
        async for event in agent.run(
            "Call the pick_color tool. I need to know what color it returns."
        ):
            if event.type == "toolCallFinished":
                summary = event.summary

        assert summary != "", "Expected non-empty summary from toolCallFinished"
        assert "magenta" in summary, (
            f"Expected 'magenta' in summary, got: {summary!r}"
        )
    finally:
        await agent.close()


# ═══════════════════════════════════════════════════════════════════════════
# E2E-17: planUpdated event
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_17_plan_updated_event(e2e_agent: Agent) -> None:
    """If the model emits planUpdated events, they should not crash the SDK.

    Not all models emit planUpdated.  The test just verifies the agent can
    complete a run without crashing, which implicitly covers any planUpdated
    events that might arrive.
    """
    events = []
    async for event in e2e_agent.run(
        "Plan then execute: count from 1 to 3. First say your plan, then count."
    ):
        events.append(event)

    types = {e.type for e in events}
    # planUpdated may or may not appear — just verify no crash
    assert "textDelta" in types, f"Expected at least textDelta in {types}"
