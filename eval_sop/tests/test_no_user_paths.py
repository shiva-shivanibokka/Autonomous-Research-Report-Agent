"""Committed evidence/results must not leak local user-profile paths."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USER_PATH = re.compile(r"[A-Za-z]:(?:\x5c|/)+Users(?:\x5c|/)+(?!<)[^\x5c/\s]+(?:\x5c|/)", re.IGNORECASE)


def test_no_user_profile_paths_in_committed_outputs():
    files = [*(ROOT / "eval_sop" / "evidence").glob("*.txt"),
             *(ROOT / "eval_sop" / "results").glob("*.json*"), ROOT / "RESULTS.md"]
    hits = [f.name for f in files if USER_PATH.search(f.read_text(encoding="utf-8", errors="replace"))]
    assert hits == []
