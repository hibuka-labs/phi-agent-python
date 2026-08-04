"""Manage the ``phi serve`` subprocess lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .protocol import ProtocolError, ProtocolStream

_log = logging.getLogger("phi-agent")


def _load_dotenv() -> dict[str, str]:
    """Load ``.env`` from the current working directory.

    Returns a dict of ``KEY=value`` pairs.  Does NOT modify ``os.environ``
    — the caller injects them into the subprocess.
    """
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        return {}

    vars_: dict[str, str] = {}
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            vars_[key] = val
    _log.info(f"loaded {len(vars_)} vars from {env_file}")
    return vars_


def _find_phi_binary() -> str:
    """Locate the ``phi`` binary.

    Resolution order:
    1. ``PHI_PATH`` environment variable
    2. ``<package>/bin/phi`` (bundled binary)
    3. ``phi`` on ``$PATH``

    Raises ``FileNotFoundError`` if no binary is found, or if the resolved
    path exists but is not executable.
    """
    env_path = os.environ.get("PHI_PATH")
    if env_path:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f"PHI_PATH={env_path}: file not found")
        if not os.access(env_path, os.X_OK):
            raise PermissionError(f"PHI_PATH={env_path}: file is not executable")
        _log.info(f"phi binary from PHI_PATH: {env_path}")
        return env_path

    bundled = Path(__file__).resolve().parent.parent.parent / "bin" / "phi"
    if bundled.is_file() and os.access(str(bundled), os.X_OK):
        _log.info(f"phi binary bundled: {bundled}")
        return str(bundled)

    which = shutil.which("phi")
    if which:
        _log.info(f"phi binary from PATH: {which}")
        return which

    _log.error("phi binary not found")
    raise FileNotFoundError(
        "phi binary not found.  Install phi-agent (Rust) or set PHI_PATH."
    )


class PhiProcess:
    """Manages a ``phi serve`` subprocess.

    Usage::

        async with PhiProcess() as proc:
            await proc.handshake()
            ...
    """

    def __init__(self, phi_path: str | None = None, env: dict[str, str] | None = None) -> None:
        self._phi_path = phi_path or _find_phi_binary()
        self._env = env or {}
        self._process: asyncio.subprocess.Process | None = None
        self._stream: ProtocolStream | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._protocol_version: int = 0
        self._server_version: str = ""

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def stream(self) -> ProtocolStream:
        if self._stream is None:
            raise RuntimeError("phi process not started")
        return self._stream

    @property
    def protocol_version(self) -> int:
        return self._protocol_version

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn ``phi serve`` and open the NDJSON stream."""
        # Load .env from current directory
        env_vars = _load_dotenv()

        # Merge: parent env + .env + SDK overrides
        merged_env = {**os.environ, **env_vars, **self._env}

        _log.info(f"spawning: {self._phi_path} serve")
        if self._env:
            _log.info(f"extra env: {list(self._env.keys())}")

        self._process = await asyncio.create_subprocess_exec(
            self._phi_path,
            "serve",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        _log.info(f"phi serve started, pid={self._process.pid}")
        self._stream = ProtocolStream(
            reader=self._process.stdout,  # type: ignore[arg-type]
            writer=self._process.stdin,  # type: ignore[arg-type]
        )

        # Drain stderr in background so the OS pipe buffer never fills
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        """Read stderr from phi serve and log it — prevents pipe buffer deadlock."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    _log.debug(f"[phi stderr] {text}")
        except Exception:
            pass

    async def handshake(self) -> dict[str, Any]:
        """Wait for the ``hello`` message and check protocol compatibility."""
        hello = await self.stream.recv()
        if hello.get("type") != "hello":
            raise ProtocolError(f"expected hello, got {hello.get('type')}")
        self._protocol_version = hello.get("protocol_version", 0)
        self._server_version = hello.get("server_version", "unknown")
        _log.info(f"handshake ok: protocol=v{self._protocol_version}, server={self._server_version}")
        if self._protocol_version != 1:
            _log.error(f"unsupported protocol version: {self._protocol_version}")
            raise ProtocolError(
                f"unsupported protocol version {self._protocol_version}"
            )
        return hello

    async def close(self) -> None:
        """Cancel any in-flight turn and terminate the subprocess."""
        _log.info("closing phi process")

        # Cancel stderr drain first
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None

        if self._stream is not None:
            await self._stream.close()
        if self._process is not None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
                _log.info(f"phi process terminated, rc={self._process.returncode}")
            except (ProcessLookupError, TimeoutError, asyncio.TimeoutError):
                self._process.kill()
                await self._process.wait()
                _log.warning("phi process killed after timeout")

    # ── Context manager ────────────────────────────────────────────────

    async def __aenter__(self) -> "PhiProcess":
        await self.start()
        await self.handshake()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
