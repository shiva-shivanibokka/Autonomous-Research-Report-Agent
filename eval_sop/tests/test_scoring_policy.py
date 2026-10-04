"""
Scoring policy (round-2 review, decided before any paid run):
* a run is ERRORED (counted wrong in the primary analysis) only if it raised,
  the pipeline set fatal_error, or a non-pinned model was used;
* non-fatal pipeline `errors` (e.g. the Writer's "malformed JSON; raw output
  preserved", a schema mismatch) are recorded but the run is scored normally;
* a strict sensitivity analysis also counts those runs as wrong.
"""

import asyncio
import json

from eval_sop import common
from eval_sop.tests.test_repro_review import _fake_pipeline_env


def test_nonfatal_writer_error_is_scored_not_errored(monkeypatch):
    from eval_sop import run_conditions as rc

    patched = _fake_pipeline_env(monkeypatch)
    inner = common.chat

    async def malformed_writer(model, messages, *, max_tokens, agent=None, **kw):
        agent = agent or common.CURRENT_AGENT.get()
        if agent in ("writer", "writer_repair", "writer_retry"):
            return "this is not json at all", 10, 5, False
        return await inner(model, messages, max_tokens=max_tokens, agent=agent, **kw)

    monkeypatch.setattr(common, "chat", malformed_writer)
    q = {"qid": "t2", "question": "What is the test question here?", "reference": "x"}
    rec = asyncio.run(rc.run_one("d", q, 1, patched))
    assert any("malformed JSON" in e for e in rec["pipeline"]["errors"])  # precondition
    assert rec["error"] is None and rec["error_kind"] is None
    assert rec["nonfatal_errors"] and rec["strict_error"]


def test_primary_itt_scores_nonfatal_runs_and_strict_itt_does_not(tmp_path, monkeypatch):
    from eval_sop import score

    runs = tmp_path / "runs"
    for cond, err, nonfatal, correct in [
        ("search1", None, [], False),
        ("pipeline_r2", None, ["Writer returned malformed JSON; raw output preserved."], True),
    ]:
        d = runs / cond
        d.mkdir(parents=True)
        (d / "q1__s1.json").write_text(json.dumps({
            "condition": cond, "qid": "q1", "seed": 1, "question": "q?", "reference": "Paris",
            "final_answer": "Paris" if correct else "Rome", "error": err, "error_kind": None,
            "nonfatal_errors": nonfatal, "strict_error": bool(nonfatal), "claims": [],
            "telemetry": {}}), encoding="utf-8")
    monkeypatch.setattr(score, "RUNS", runs)
    monkeypatch.setattr(score, "RES", tmp_path / "res")
    monkeypatch.setattr(score, "JUDGED", tmp_path / "res" / "j.jsonl")
    score.analyze()
    summ = json.loads((tmp_path / "res" / "summary.json").read_text(encoding="utf-8"))
    primary = summ["mcnemar_itt"]["pipeline_r2 vs search1"]["strict_match"]
    strict = summ["mcnemar_itt_strict"]["pipeline_r2 vs search1"]["strict_match"]
    assert primary["acc_b"] == 1.0 and strict["acc_b"] == 0.0
