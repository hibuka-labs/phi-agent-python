"""UT-25 → UT-27: Agent config resolution — constructor args, env fallback, .env loading.

Pure Python unit tests — no phi binary needed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from phi_agent import Agent
from phi_agent.process import _load_dotenv


# ═══════════════════════════════════════════════════════════════════════════
# UT-25: Constructor args take priority over env vars
# ═══════════════════════════════════════════════════════════════════════════


def test_ut_25_constructor_overrides_env_model() -> None:
    """Agent(model='x') should override LLM_MODEL env var."""
    with patch.dict(os.environ, {"LLM_MODEL": "env-model"}):
        agent = Agent(model="constructor-model")
        assert agent._model == "constructor-model"


def test_ut_25_constructor_overrides_env_api_key() -> None:
    """Agent(api_key='x') should override LLM_API_KEY env var."""
    with patch.dict(os.environ, {"LLM_API_KEY": "env-key"}):
        agent = Agent(api_key="constructor-key")
        assert agent._api_key == "constructor-key"


def test_ut_25_constructor_overrides_env_base_url() -> None:
    """Agent(base_url='x') should override LLM_BASE_URL env var."""
    with patch.dict(os.environ, {"LLM_BASE_URL": "https://env.example.com"}):
        agent = Agent(base_url="https://ctor.example.com")
        assert agent._base_url == "https://ctor.example.com"


def test_ut_25_falsy_constructor_falls_back_to_env() -> None:
    """Agent(api_key='') should fall back to env var (empty string is falsy)."""
    with patch.dict(os.environ, {"LLM_API_KEY": "env-key"}):
        agent = Agent(api_key="")
        assert agent._api_key == "env-key"


# ═══════════════════════════════════════════════════════════════════════════
# UT-26: Env fallback when no constructor arg
# ═══════════════════════════════════════════════════════════════════════════


def test_ut_26_env_fallback_model() -> None:
    """Without constructor model, LLM_MODEL env var should be used."""
    with patch.dict(os.environ, {"LLM_MODEL": "env-model"}):
        agent = Agent()
        assert agent._model == "env-model"


def test_ut_26_env_fallback_default() -> None:
    """Without constructor or env var, default 'gpt-4o' should be used."""
    with patch.dict(os.environ, {}, clear=True):
        agent = Agent()
        assert agent._model == "gpt-4o"


def test_ut_26_env_fallback_api_key() -> None:
    """Without constructor api_key, LLM_API_KEY env var should be used."""
    with patch.dict(os.environ, {"LLM_API_KEY": "env-key-123"}):
        agent = Agent()
        assert agent._api_key == "env-key-123"


def test_ut_26_env_fallback_api_key_none() -> None:
    """Without constructor or env, api_key should be None."""
    with patch.dict(os.environ, {}, clear=True):
        agent = Agent()
        assert agent._api_key is None


# ═══════════════════════════════════════════════════════════════════════════
# UT-27: .env loading
# ═══════════════════════════════════════════════════════════════════════════


def test_ut_27_load_dotenv_basic() -> None:
    """_load_dotenv should parse KEY=value pairs from a .env file."""
    content = "LLM_API_KEY=sk-test-123\nLLM_MODEL=gpt-4o\n# comment\n\nLLM_BASE_URL=https://example.com/v1\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / ".env"
        env_file.write_text(content)

        with patch("phi_agent.process.Path.cwd", return_value=Path(tmpdir)):
            vars_ = _load_dotenv()
            assert vars_["LLM_API_KEY"] == "sk-test-123"
            assert vars_["LLM_MODEL"] == "gpt-4o"
            assert vars_["LLM_BASE_URL"] == "https://example.com/v1"


def test_ut_27_load_dotenv_quotes_stripped() -> None:
    """Values wrapped in quotes should have quotes stripped."""
    content = 'LLM_API_KEY="sk-quoted"\nLLM_MODEL=\'gpt-4o\'\n'
    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / ".env"
        env_file.write_text(content)

        with patch("phi_agent.process.Path.cwd", return_value=Path(tmpdir)):
            vars_ = _load_dotenv()
            assert vars_["LLM_API_KEY"] == "sk-quoted"
            assert vars_["LLM_MODEL"] == "gpt-4o"


def test_ut_27_load_dotenv_no_file() -> None:
    """When .env doesn't exist, return empty dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("phi_agent.process.Path.cwd", return_value=Path(tmpdir)):
            vars_ = _load_dotenv()
            assert vars_ == {}


def test_ut_27_load_dotenv_injected_into_subprocess() -> None:
    """_load_dotenv does NOT modify os.environ — caller injects vars."""
    with patch.dict(os.environ, {}, clear=True):
        content = "LLM_API_KEY=sk-test\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(content)

            with patch("phi_agent.process.Path.cwd", return_value=Path(tmpdir)):
                vars_ = _load_dotenv()
                assert vars_["LLM_API_KEY"] == "sk-test"
                # os.environ should NOT be modified
                assert "LLM_API_KEY" not in os.environ
