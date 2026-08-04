"""Tests for the phi-agent Python SDK protocol layer.

These tests validate that the protocol messages are correctly
serialized / deserialized.  They do NOT require a running phi binary.
"""

from phi_agent.events import Event
from phi_agent.tool import _build_parameters_schema, _type_to_json_schema, tool


class TestEvent:
    def test_text_delta(self):
        raw = {
            "runtimeEventType": "textDelta",
            "sessionId": {"id": 1, "externalId": "abc"},
            "text": "Hello, world!",
        }
        event = Event(type=raw["runtimeEventType"], data=raw)
        assert event.type == "textDelta"
        assert event.text == "Hello, world!"
        assert event.session_id == {"id": 1, "externalId": "abc"}

    def test_tool_call_started(self):
        raw = {
            "runtimeEventType": "toolCallStarted",
            "sessionId": {"id": 1},
            "toolName": "search",
            "argsJson": '{"query":"weather"}',
        }
        event = Event(type=raw["runtimeEventType"], data=raw)
        assert event.type == "toolCallStarted"
        assert event.tool_name == "search"
        assert event.args_json == '{"query":"weather"}'

    def test_unknown_event_type_is_not_crashing(self):
        """Forward-compat: new event types should not break the SDK."""
        raw = {
            "runtimeEventType": "someFutureEvent",
            "sessionId": {"id": 1},
            "newField": 42,
        }
        event = Event(type=raw["runtimeEventType"], data=raw)
        assert event.type == "someFutureEvent"
        # Unknown fields are silently carried in data, no exception


class TestToolDecorator:
    async def test_simple_tool(self):
        @tool
        async def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}"

        assert greet.name == "greet"
        assert greet.description == "Say hello."
        assert greet.parameters["properties"]["name"] == {"type": "string"}
        assert greet.parameters["required"] == ["name"]
        result = await greet.func(name="World")
        assert result == "Hello, World"

    async def test_custom_name_and_description(self):
        @tool(name="hello", description="Greet someone")
        async def greet(name: str) -> str:
            return f"Hi, {name}"

        assert greet.name == "hello"
        assert greet.description == "Greet someone"


class TestJsonSchema:
    def test_basic_types(self):
        assert _type_to_json_schema(str) == {"type": "string"}
        assert _type_to_json_schema(int) == {"type": "integer"}
        assert _type_to_json_schema(float) == {"type": "number"}
        assert _type_to_json_schema(bool) == {"type": "boolean"}

    def test_list_type(self):
        schema = _type_to_json_schema(list[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_build_parameters(self):
        async def fn(a: str, b: int = 10) -> str: ...
        params = _build_parameters_schema(fn)
        assert params["type"] == "object"
        assert params["properties"]["a"] == {"type": "string"}
        assert params["properties"]["b"] == {"type": "integer"}
        assert params["required"] == ["a"]
