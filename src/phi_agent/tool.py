"""Tool definition — the ``@tool`` decorator and ``ToolOutput`` result type."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolOutput:
    """Returned by a tool to give the agent structured output."""

    summary: str
    """Human-readable summary shown to the LLM."""

    raw: dict[str, Any] | None = None
    """Optional structured data (available to the caller)."""


# ── JSON Schema helpers ────────────────────────────────────────────────

_PY_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _type_to_json_schema(typ: type) -> dict[str, Any]:
    """Best-effort conversion of a Python type hint to a JSON Schema."""
    origin = getattr(typ, "__origin__", None)
    if origin is list:
        args = getattr(typ, "__args__", ())
        item_schema = _type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}
    json_type = _PY_TO_JSON_TYPE.get(typ)
    if json_type:
        return {"type": json_type}
    return {"type": "string"}


def _build_parameters_schema(
    func: Callable[..., Any],
) -> dict[str, Any]:
    """Build a JSON Schema ``parameters`` object from a function signature."""
    sig = inspect.signature(func)
    required: list[str] = []
    properties: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        properties[name] = _type_to_json_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ── Decorator ──────────────────────────────────────────────────────────

def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
):
    """Decorator that turns an async function into a phi-agent tool.

    The function signature is inspected to auto-generate the JSON Schema
    that the LLM sees.  Type annotations are mapped to JSON types where
    possible.

    Usage::

        @tool
        async def search(query: str, limit: int = 10) -> str:
            '''Search the web.'''
            ...

        @tool(name="get_weather", description="Look up the weather")
        async def weather(city: str) -> str:
            ...
    """

    def _decorator(fn: Callable[..., Any]) -> RegisteredTool:
        tool_name = name or fn.__name__
        tool_desc = description or _extract_docstring(fn)
        schema = _build_parameters_schema(fn)
        return RegisteredTool(
            name=tool_name,
            description=tool_desc,
            parameters=schema,
            func=fn,
        )

    if func is not None:
        return _decorator(func)
    return _decorator


def _extract_docstring(fn: Callable[..., Any]) -> str:
    doc = inspect.getdoc(fn)
    if doc:
        return doc.split("\n")[0]
    return fn.__name__


# ── Registered tool ────────────────────────────────────────────────────

@dataclass
class RegisteredTool:
    """Internal representation of a user-defined tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
