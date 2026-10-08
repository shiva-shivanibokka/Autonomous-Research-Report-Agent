"""
Round-4 review: scripts/record_demo_run.py spends on claude-sonnet-4-5 with no
ledger, cap or lock, using whatever ANTHROPIC_API_KEY is in the environment or
the repo .env. It must not be possible to start it by accident, and the
runbook must say the paid key goes only in the --env-file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "record_demo_run.py"


def _run(args, key="sk-ant-looks-real"):
    env = {**os.environ, "ANTHROPIC_API_KEY": key, "TAVILY_API_KEY": "tvly-x"}
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO, env=env,  # noqa: S603
                          capture_output=True, text=True, timeout=300)


def test_refuses_without_the_opt_in_flag():
    r = _run([])
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "--i-want-to-spend-real-money" in out and "not metered" in out.lower()


def test_dry_run_is_not_gated_by_the_flag():
    """--dry-run spends nothing, so it must get past the opt-in gate. (It then
    stops on this worktree's missing .env, which is not the gate.)"""
    out = _run(["--dry-run"])
    assert "--i-want-to-spend-real-money" not in out.stdout + out.stderr


def test_runbook_warns_that_the_key_goes_only_in_the_env_file():
    results = (REPO / "RESULTS.md").read_text(encoding="utf-8").lower()
    assert "--env-file" in results
    assert "never in the repo" in results and "never exported" in results
    assert "record_demo_run.py" in results
