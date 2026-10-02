"""
Offline smoke test of the eval harness: all four conditions + scoring, with a
scripted fake LLM, fake search and fake pages. No network, no models, no
credits. Writes only under eval_sop/cache/smoke/ (gitignored).

  python -m eval_sop.smoke_test

It checks plumbing (state capture across rounds, claim export, telemetry,
analysis), not quality. Real numbers need a real model; see RESULTS.md.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_SMOKE_ROOT = Path(__file__).resolve().parent / "cache" / "smoke"  # gitignored
_SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
TMP = tempfile.mkdtemp(prefix="run_", dir=_SMOKE_ROOT)
os.environ["SOP_OUT_DIR"] = TMP

from eval_sop import common  # noqa: E402

CRITIC_CALLS = [0]


async def fake_chat(model, messages, *, max_tokens, seed=None, temperature=None,
                    num_ctx=None, agent=None):
    agent = agent or common.CURRENT_AGENT.get()
    user = messages[-1]["content"]
    if agent == "orchestrator":
        body = {"sub_questions": ["sub q one", "sub q two"], "reasoning": "r"}
    elif agent == "analyst":
        r = CRITIC_CALLS[0]
        body = {"claims": [
            {"text": f"Claim r{r} for {user[22:32]}", "source_urls": ["https://a.example/1"],
             "supporting_sources": 3, "contradicting_sources": 0},
            {"text": f"Weak claim r{r} {user[22:32]}", "source_urls": ["https://b.example/1"],
             "supporting_sources": 1, "contradicting_sources": 1, "contradiction_detail": "x"}],
            "contradictions_found": 1, "avg_confidence_score": 0.5}
    elif agent == "critic":
        CRITIC_CALLS[0] += 1
        first = "Research round: 1 of" in user
        body = {"flagged_claims": [{"claim_text": "Weak claim", "reason": "r",
                                    "re_search_query": "verify weak"}] if first else [],
                "overall_quality_score": 0.5, "coverage_score": 0.6,
                "source_diversity_score": 0.7, "contradiction_rate": 0.3,
                "needs_more_research": first, "critic_notes": "n"}
    elif agent == "fact_checker":
        body = {"verdict": "inconclusive", "evidence": "e", "confidence": "low", "explanation": "x"}
    elif agent == "writer":
        body = {"title": "T", "executive_summary": "The answer is Jane Ballou.", "key_findings": ["k"],
                "detailed_sections": {"A": "a"}, "confidence_assessment": "c", "limitations": "l"}
    else:  # closed_book, search1, eval_answer_extraction, judges
        if "Ground-truth answer" in user:
            return "CORRECT", 10, 1, False
        return "- A fact [1]\n- Another fact [2]\nFinal answer: Jane Ballou", 10, 10, False
    return json.dumps(body), 100, 50, False


def main():
    patched = common.install()
    common.chat = fake_chat
    import eval_sop.run_conditions as rc
    import eval_sop.score as score
    from agents.schemas import ScrapedPage, SearchResult

    async def fake_search(query, *, max_results=8, **_):
        return [SearchResult(url=f"https://{h}.example/1", title="t", snippet="s",
                             relevance_score=0.9, source_domain=f"{h}.example") for h in ("a", "b")]

    async def fake_scrape(url, title=""):
        return ScrapedPage(url=url, title="t", content="page text about Jane Ballou", word_count=5)

    import agents.fact_checker_agent as fca
    import agents.search_agent as sa
    import agents.tools.scraper_tool as st
    sa.tavily_search = fca.tavily_search = fake_search
    st.scrape_page = fake_scrape
    patched.search = fake_search
    rc.common.install = lambda: patched

    class A:
        conditions, seeds, n_frames, open, only, force = "a,b,c,d", "1", 2, False, "", False
    asyncio.run(rc.main_async(A))
    recs = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in (rc.RUNS).glob("*/*.json")}
    assert set(recs) == {"closed_book", "search1", "pipeline_r1", "pipeline_r2"}, recs.keys()
    for name, r in recs.items():
        assert not r["error"], (name, r["error"])
    r2 = recs["pipeline_r2"]
    assert r2["pipeline"]["rounds_run"] == 2
    assert set(r2["rounds"]) == {0, 1} or set(r2["rounds"]) == {"0", "1"}
    assert recs["pipeline_r1"]["pipeline"]["fact_check_results"], "r1 should fact-check flags"
    # round-1 claims carried into the final claim set (the fix in agents/graph.py)
    assert any(c.get("round") == 0 for c in r2["claims"]), r2["claims"]
    score.analyze()
    summ = json.loads((score.RES / "summary.json").read_text(encoding="utf-8"))
    cv = summ["critic_vs_computed"]
    assert cv["n_critic_calls"] >= 3, cv
    print("SMOKE OK", TMP)
    print(json.dumps(cv, indent=1)[:800])


if __name__ == "__main__":
    sys.exit(main())
