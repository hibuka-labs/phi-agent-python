"""E2E-27 → E2E-29: Python version compatibility.

Validates that the SDK's type-inspection code (``inspect.signature``,
``getattr(typ, "__origin__")``, type-annotation mapping) works correctly.
These tests are designed to run in a CI matrix across Python 3.10, 3.11, 3.12.

They don't need phi serve or an LLM — they exercise the pure-Python tool
and event layers, which are the compatibility-sensitive parts.

NOTE: This file intentionally does NOT use ``from __future__ import annotations``
because the tests need runtime-resolvable type annotations for compatibility
verification.  Lazy annotations (PEP 563) turn all type hints into strings,
which defeats the ``inspect.signature`` tests below.
"""

import sys
from typing import Optional

import pytest

from phi_agent import tool
from phi_agent.events import Event
from phi_agent.tool import ToolOutput


# ═══════════════════════════════════════════════════════════════════════════
# E2E-27→29: Type annotation compatibility (all versions)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_e2e_27_29_type_inspection_works() -> None:
    """Verify ``inspect.signature`` and type annotation mapping work on
    the current Python version (covers E2E-27, E2E-28, E2E-29)."""
    version_info = sys.version_info[:2]
    assert version_info >= (3, 10), f"Expected >=3.10, got {version_info}"

    # ── Test 1: inspect.signature on a tool function ──

    @tool
    async def sample_tool(
        name: str,
        count: int = 1,
        active: bool = False,
        tags: list[str] | None = None,
    ) -> str:
        """A sample tool for compatibility testing."""
        return f"Hello {name}"

    # inspect.signature should parse correctly across versions
    assert sample_tool.name == "sample_tool"
    assert sample_tool.description == "A sample tool for compatibility testing."

    params = sample_tool.parameters
    assert params["type"] == "object"

    # str → "string"
    assert params["properties"]["name"] == {"type": "string"}
    # int → "integer"
    assert params["properties"]["count"] == {"type": "integer"}
    # bool → "boolean"
    assert params["properties"]["active"] == {"type": "boolean"}

    # required: name is required, others have defaults
    assert "name" in params["required"]
    assert "count" not in params["required"]

    # ── Test 2: getattr(typ, "__origin__") for complex types ──
    # list[str] | None is a UnionType — SDK degrades to "string" without
    # crashing (per UT-20: complex generics degrade gracefully).
    tags_schema = params["properties"].get("tags")
    assert tags_schema is not None
    # UnionType (list[str] | None) falls back to "string" — acceptable
    assert tags_schema.get("type") in ("array", "string"), (
        f"Expected array or string (degraded), got {tags_schema}"
    )


@pytest.mark.asyncio
async def test_e2e_27_29_optional_type() -> None:
    """Optional[str] and similar typing constructs should not crash the SDK."""

    @tool
    async def with_optional(
        required_str: str,
        optional_str: Optional[str] = None,
    ) -> str:
        """Tool with Optional type."""
        return required_str

    # Optional[str] should not crash — it may map to "string" (degraded)
    # but must not raise an exception
    optional_schema = with_optional.parameters["properties"]["optional_str"]
    assert "type" in optional_schema, (
        f"Expected type key in schema: {optional_schema}"
    )
    # required_str is required, optional_str is not
    assert "optional_str" not in with_optional.parameters["required"]


@pytest.mark.asyncio
async def test_e2e_27_29_event_handling() -> None:
    """Event accessors work correctly — unknown fields don't crash."""
    # textDelta event
    e = Event(type="textDelta", data={"text": "hello world", "sessionId": {"id": 1}})
    assert e.text == "hello world"
    assert e.tool_name == ""  # wrong type, returns ""
    assert e.args_json == "{}"  # wrong type, returns "{}"
    assert e.session_id == {"id": 1}

    # toolCallStarted event
    e2 = Event(type="toolCallStarted", data={
        "toolName": "search",
        "argsJson": '{"query":"test"}',
    })
    assert e2.tool_name == "search"
    assert e2.args_json == '{"query":"test"}'
    assert e2.text == ""

    # toolCallFinished event
    e3 = Event(type="toolCallFinished", data={
        "toolName": "search",
        "summary": "Found 3 results",
    })
    assert e3.summary == "Found 3 results"

    # Unknown event type — forward compatible
    e4 = Event(type="futureEventType", data={"newField": 42})
    assert e4.type == "futureEventType"
    assert e4.text == ""

    # Missing fields — should not crash
    e5 = Event(type="textDelta")
    assert e5.text == ""


@pytest.mark.asyncio
async def test_e2e_27_29_tool_output() -> None:
    """ToolOutput dataclass works correctly."""
    # Basic
    out = ToolOutput(summary="Done")
    assert out.summary == "Done"
    assert out.raw is None

    # With raw
    out2 = ToolOutput(summary="Done", raw={"count": 42})
    assert out2.raw == {"count": 42}
