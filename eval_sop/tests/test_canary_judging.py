"""
Canary output must (1) stay out of the main analysis and (2) be judgeable on
its own, so the local correctness judges can be checked right after the
~$0.61 canary and before the main paid run (round-2 review).
"""

import asyncio
import json
from types import SimpleNamespace

from eval_sop import common


def test_canary_runs_are_separate_and_judgeable(tmp_path, monkeypatch):
    from eval_sop import run_conditions as rc
    from eval_sop import score
    from eval_sop.tests.test_repro_review import _fake_pipeline_env

    patched = _fake_pipeline_env(monkeypatch)
    import agents.search_agent as sa

    patched.search = sa.tavily_search  # the fake installed above: no network in tests
    monkeypatch.setattr(rc, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(rc, "CANARY_RUNS", tmp_path / "canary_runs")
    args = SimpleNamespace(backend="ollama")
    assert asyncio.run(rc._run_canary(args, patched, ["a", "b"])) in (0, 1)

    monkeypatch.setattr(score, "RUNS", tmp_path / "runs")
    assert score.load_runs() == []  # canary never mixes into the main analysis

    judged = []

    async def fake_judge(model, question, reference, prediction):
        judged.append((model, reference, prediction))
        return {"raw": "CORRECT", "correct": True}

    monkeypatch.setattr(score, "judge_correct", fake_judge)
    monkeypatch.setattr(score, "RUNS", tmp_path / "canary_runs")
    monkeypatch.setattr(score, "RES", tmp_path / "canary_res")
    monkeypatch.setattr(score, "JUDGED", tmp_path / "canary_res" / "j.jsonl")
    asyncio.run(score.judge_all())
    rows = [json.loads(x) for x in open(tmp_path / "canary_res" / "j.jsonl", encoding="utf-8")]
    assert {r["judge"] for r in rows} == set(common.JUDGE_MODELS)
    assert len(rows) == 2 * len(common.JUDGE_MODELS)
