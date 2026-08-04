"""UT-21 → UT-24: _find_phi_binary — path resolution, existence, executability, symlinks.

Pure Python unit tests — no phi binary needed.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from phi_agent.process import _find_phi_binary


# ═══════════════════════════════════════════════════════════════════════════
# UT-21: PHI_PATH env var takes priority
# ═══════════════════════════════════════════════════════════════════════════


def test_ut_21_phi_path_env_priority() -> None:
    """PHI_PATH env var should be checked first and take priority."""
    with tempfile.NamedTemporaryFile(suffix="-phi", delete=False) as f:
        f.write(b"#!/bin/sh\necho ok\n")
        f.flush()
        os.chmod(f.name, 0o755)
        tmp_path = f.name

    try:
        with patch.dict(os.environ, {"PHI_PATH": tmp_path}):
            result = _find_phi_binary()
            assert result == tmp_path, f"Expected {tmp_path}, got {result}"
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# UT-22: File not found → FileNotFoundError with actionable message
# ═══════════════════════════════════════════════════════════════════════════


def test_ut_22_file_not_found_raises() -> None:
    """When PHI_PATH points to a nonexistent file, raise FileNotFoundError."""
    with patch.dict(os.environ, {"PHI_PATH": "/nonexistent/path/phi-xyz"}):
        with pytest.raises(FileNotFoundError, match="file not found"):
            _find_phi_binary()


def test_ut_22_no_binary_anywhere() -> None:
    """When no binary is found via any method, raise FileNotFoundError
    with an actionable message."""
    with patch.dict(os.environ, {}, clear=True):
        # Ensure PHI_PATH is not set
        with patch("phi_agent.process.shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="phi binary not found"):
                _find_phi_binary()


# ═══════════════════════════════════════════════════════════════════════════
# UT-23: Non-executable file → PermissionError
# ═══════════════════════════════════════════════════════════════════════════


def test_ut_23_non_executable_file() -> None:
    """When PHI_PATH points to a non-executable file, raise PermissionError."""
    with tempfile.NamedTemporaryFile(suffix="-phi", delete=False) as f:
        f.write(b"not executable\n")
        f.flush()
        os.chmod(f.name, 0o644)  # rw-r--r--, not executable
        tmp_path = f.name

    try:
        with patch.dict(os.environ, {"PHI_PATH": tmp_path}):
            with pytest.raises(PermissionError, match="not executable"):
                _find_phi_binary()
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# UT-24: Symlink — no infinite loop
# ═══════════════════════════════════════════════════════════════════════════


def test_ut_24_symlink_loop_no_hang() -> None:
    """Self-referencing symlink should not cause an infinite loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        link_path = os.path.join(tmpdir, "phi-loop")
        # Create a symlink that points to itself
        os.symlink(link_path, link_path)

        with patch.dict(os.environ, {"PHI_PATH": link_path}):
            # Self-referencing symlink: isfile() returns True for the
            # symlink node itself (it exists), but os.access(..., X_OK)
            # checks the link target which loops → should either succeed
            # quickly or raise.  Either way, must not hang.
            import signal

            def _handler(signum, frame):
                raise TimeoutError("_find_phi_binary hung on symlink loop")

            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(2)  # 2-second timeout

            try:
                # Self-symlink is a file (the symlink node exists),
                # but it's not executable (points to itself).
                # Should raise FileNotFoundError or PermissionError quickly.
                try:
                    _find_phi_binary()
                except (FileNotFoundError, PermissionError, OSError):
                    pass  # Expected — didn't hang
            except TimeoutError:
                pytest.fail("_find_phi_binary hung on symlink loop")
            finally:
                signal.alarm(0)  # Cancel alarm


def test_ut_24_broken_symlink_handled() -> None:
    """Broken symlink (dangling) should raise FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        link_path = os.path.join(tmpdir, "phi-broken")
        os.symlink("/nonexistent/target", link_path)

        with patch.dict(os.environ, {"PHI_PATH": link_path}):
            with pytest.raises(FileNotFoundError, match="file not found"):
                _find_phi_binary()
