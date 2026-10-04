"""Every eval_sop test writes under eval_sop/cache/tests/ (gitignored), never the real caches."""

import os
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "cache" / "tests"
_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["SOP_OUT_DIR"] = tempfile.mkdtemp(prefix="t_", dir=_ROOT)
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-not-a-real-key")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_stop_flag():
    from eval_sop import budget

    budget.reset_stop()
    yield
    budget.reset_stop()
