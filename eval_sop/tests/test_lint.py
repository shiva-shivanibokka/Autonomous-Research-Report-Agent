"""
The product CI runs `ruff check .` and `ruff format --check .` on the whole
repository. eval_sop/ is evaluation tooling, excluded from that style gate in
ruff.toml; this test keeps it free of real errors (pyflakes / syntax) instead.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_product_lint_gate_passes_and_eval_code_has_no_real_errors():
    for args in (["check", "."], ["format", "--check", "."], ["check", "eval_sop", "--isolated",
                                                               "--select", "E9,F"]):
        r = subprocess.run(["ruff", *args], cwd=REPO, capture_output=True, text=True)  # noqa: S603, S607
        assert r.returncode == 0, (args, r.stdout[-2000:])
