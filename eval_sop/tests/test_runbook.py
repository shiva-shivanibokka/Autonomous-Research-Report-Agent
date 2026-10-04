"""
Round-3 review: the documented canary command exited 2. These tests run the
EXACT commands from the RESULTS.md runbook (the fenced block after the
'<!-- runbook -->' marker) end to end, with a fake Anthropic client, fake
search/pages and a fake correctness judge. No network, no key.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from eval_sop import common
from eval_sop.tests.test_anthropic_budget import MODEL, FakeAnthropic, _router

RESULTS = Path(__file__).resolve().parents[2] / "RESULTS.md"


def runbook_commands() -> list[list[str]]:
    text = RESULTS.read_text(encoding="utf-8")
    block = text.split("<!-- runbook -->", 1)[1].split("```", 2)[1]
    cmds = []
    for line in block.splitlines():
        line = line.split("  #", 1)[0].strip()
        if line.startswith("python -m eval_sop."):
            cmds.append(shlex.split(line))
    return cmds


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    from agents.schemas import ScrapedPage, SearchResult

    env = tmp_path / "keys.env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-fake\nTAVILY_API_KEY=tvly-fake\n", encoding="utf-8")
    monkeypatch.setattr(common, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(common, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(common, "LEDGER", None)
    common._llm_cache.db.execute("DELETE FROM kv")
    common._llm_cache.db.commit()

    async def fake_search(query, *, max_results=8, **_):
        return [SearchResult(url="https://a.example/1", title="t", snippet="Paris is the capital.",
                             relevance_score=0.9, source_domain="a.example")]

    async def fake_scrape(url, title=""):
        return ScrapedPage(url=url, title="t", content="Paris is the capital of France.", word_count=6)

    # The originals that install() wraps: fakes, so the eval's real wrappers
    # (cache, credit ledger) still run.
    for name, fn in (("tavily_search", fake_search), ("scrape_page", fake_scrape)):
        monkeypatch.setitem(common._ORIGINALS, name, fn)
    common.set_anthropic_client(FakeAnthropic(router=_router))

    async def no_sleep(_):
        return None

    monkeypatch.setattr(common, "_sleep", no_sleep)

    from eval_sop import run_conditions as rc
    from eval_sop import score

    monkeypatch.setattr(rc, "RUNS", tmp_path / "out" / "runs")
    monkeypatch.setattr(rc, "CANARY_RUNS", tmp_path / "out" / "canary_runs")

    async def fake_judge(model, question, reference, prediction):
        ok = reference.lower() in (prediction or "").lower()
        return {"raw": "CORRECT" if ok else "INCORRECT", "correct": ok}

    monkeypatch.setattr(score, "judge_correct", fake_judge)
    monkeypatch.setattr(score, "support_nli", lambda: None)  # NLI needs the torch env
    yield env
    common.set_anthropic_client(None)


def _run(argv, env, monkeypatch):
    from eval_sop import run_conditions as rc
    from eval_sop import score

    argv = [str(env) if a == "PATH" else a for a in argv]
    module = argv[2]
    monkeypatch.setattr(sys, "argv", [module] + argv[3:])
    if module == "eval_sop.run_conditions":
        with pytest.raises(SystemExit) as e:
            rc.main()
        return e.value.code
    if module == "eval_sop.score":
        return score.main(argv[3:])
    if module == "eval_sop.tavily_usage":
        return "skipped (network)"
    raise AssertionError(module)


def test_documented_canary_command_exits_zero(fake_world, monkeypatch):
    """attack3 #7: '--canary' with the documented arguments used to exit 2."""
    argv = shlex.split("python -m eval_sop.run_conditions --backend anthropic --env-file PATH "
                       "--conditions a,b,d --n-frames 30 --usd-cap 8 --canary")
    assert _run(argv, fake_world, monkeypatch) == 0


def test_every_runbook_command_runs_end_to_end(fake_world, monkeypatch):
    cmds = runbook_commands()
    assert len(cmds) >= 6, cmds
    results = [(" ".join(c[2:5]), _run(c, fake_world, monkeypatch)) for c in cmds]
    for name, code in results:
        assert code in (0, None, "skipped (network)"), results
    # the main run really produced paired runs and analysis, on the pinned model
    out = common.OUT_DIR
    assert len(list((out / "runs" / "pipeline_r2").glob("*.json"))) == 30
    assert (out / "results" / "summary.json").exists()
    assert (out / "results" / "usd_ledger.jsonl").exists()
    assert common.LEDGER.spent() <= 8.0
    assert {r[0] for r in common.LEDGER.db.execute("SELECT model FROM calls")} == {MODEL}
