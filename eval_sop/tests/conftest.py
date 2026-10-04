"""Every eval_sop test writes under eval_sop/cache/tests/ (gitignored), never the real caches."""

import os
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "cache" / "tests"
_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["SOP_OUT_DIR"] = tempfile.mkdtemp(prefix="t_", dir=_ROOT)
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-not-a-real-key")

# The real state directory (ledgers + run lock) lives under %LOCALAPPDATA%;
# tests must never touch it.
from eval_sop import common  # noqa: E402

common.STATE_DIR = Path(os.environ["SOP_OUT_DIR"]) / "state"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_stop_flag():
    from eval_sop import budget

    budget.reset_stop()
    yield
    budget.reset_stop()


@pytest.fixture(autouse=True)
def _no_real_tavily(monkeypatch):
    """Any test that reaches the real Tavily client fails instead of hitting the network."""
    import agents.tools.search_tool as st

    def refuse(*a, **k):
        raise AssertionError("test tried to construct a real Tavily client")

    monkeypatch.setattr(st, "AsyncTavilyClient", refuse)
    monkeypatch.setattr(st, "_client", None)
