"""Protocol event types — thin wrappers around the NDJSON wire format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """A single event from the agent runtime.

    The ``type`` field corresponds to ``RuntimeEvent.runtimeEventType`` on the
    Rust side.  Unknown fields in the raw JSON are silently ignored so that the
    SDK stays compatible when the Rust kernel adds new fields.
    """

    type: str
    """Event type tag, e.g. ``"textDelta"``, ``"toolCallStarted"``."""

    data: dict[str, Any] = field(default_factory=dict)
    """Raw event payload (the full JSON object from the wire)."""

    # ── Convenience accessors ──────────────────────────────────────────

    @property
    def text(self) -> str:
        """Shortcut for ``textDelta`` / ``thoughtDelta`` text."""
        return self.data.get("text", "")

    @property
    def tool_name(self) -> str:
        """Shortcut for ``toolCallStarted`` / ``toolCallFinished`` tool name.

        Handles both snake_case (wire format) and camelCase (legacy).
        """
        return self.data.get("tool_name") or self.data.get("toolName", "")

    @property
    def args_json(self) -> str:
        """Shortcut for ``toolCallStarted`` args (JSON string).

        Handles both snake_case (wire format) and camelCase (legacy).
        """
        return self.data.get("args_json") or self.data.get("argsJson", "{}")

    @property
    def summary(self) -> str:
        """Shortcut for ``toolCallFinished`` summary text."""
        return self.data.get("summary", "")

    @property
    def session_id(self) -> dict[str, Any] | None:
        """The session ID object ``{id, externalId}`` from the event.

        Handles both snake_case (wire format) and camelCase (legacy).
        """
        return self.data.get("session_id") or self.data.get("sessionId")

    def __repr__(self) -> str:
        extra = ""
        if self.type in ("textDelta", "thoughtDelta"):
            extra = f" text={self.text[:40]!r}"
        elif self.type == "toolCallStarted":
            extra = f" tool={self.tool_name!r}"
        elif self.type == "toolCallFinished":
            extra = f" tool={self.tool_name!r}"
        return f"Event(type={self.type!r}{extra})"
