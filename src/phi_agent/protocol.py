"""NDJSON line-protocol reader / writer over stdio.

Logs are written to ``~/.phi-agent/sdk-YYYY-MM-DD.log`` via the standard
``logging`` module.  Set ``PHI_LOG_LEVEL=DEBUG`` to see protocol messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Logger setup ───────────────────────────────────────────────────────

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("phi-agent")
    logger.setLevel(os.environ.get("PHI_LOG_LEVEL", "INFO"))

    log_dir = Path.home() / ".phi-agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_dir / f"sdk-{datetime.now().strftime('%Y-%m-%d')}.log"

    handler = logging.FileHandler(str(fname))
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger

_log = _setup_logger()


class ProtocolError(Exception):
    """Raised when the protocol stream is malformed or the child exits."""


class ProtocolStream:
    """Reads / writes NDJSON messages over a subprocess's stdin and stdout.

    One line = one JSON object.  stderr is reserved for Rust-side tracing logs
    and is not part of the protocol.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._write_lock = asyncio.Lock()

    # ── Read ───────────────────────────────────────────────────────────

    async def recv(self) -> dict[str, Any]:
        """Read the next JSON line from stdout."""
        line = await self._reader.readline()
        if not line:
            _log.error("phi process closed stdout unexpectedly")
            raise ProtocolError("phi process closed stdout unexpectedly")
        try:
            msg = json.loads(line.decode("utf-8"))
            kind = msg.get("type", "?")
            # Truncate event payloads for readability
            if kind == "event":
                et = msg.get("runtimeEventType", msg.get("event", {}).get("runtimeEventType", "?"))
                _log.debug(f"← event seq={msg.get('seq','?')} type={et}")
            else:
                _log.info(f"← {kind} {_summarize(msg)}")
            return msg
        except json.JSONDecodeError as exc:
            _log.error(f"invalid JSON: {exc}")
            raise ProtocolError(f"invalid JSON from phi: {exc}") from exc

    # ── Write ──────────────────────────────────────────────────────────

    async def send(self, msg: dict[str, Any]) -> None:
        """Write one JSON line to stdin (thread-safe)."""
        kind = msg.get("type", "?")
        _log.info(f"→ {kind} {_summarize(msg)}")
        payload = json.dumps(msg, ensure_ascii=False) + "\n"
        async with self._write_lock:
            self._writer.write(payload.encode("utf-8"))
            await self._writer.drain()

    # ── Close ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the write side of the stream."""
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass


def _summarize(msg: dict[str, Any]) -> str:
    """Short summary of a protocol message for logging."""
    parts = []
    for k in ("name", "query", "session_id", "internal_id", "outcome", "call_id", "code", "message", "ok"):
        if k in msg:
            v = msg[k]
            if isinstance(v, str) and len(v) > 60:
                v = v[:57] + "..."
            parts.append(f"{k}={v!r}")
    return " ".join(parts)
