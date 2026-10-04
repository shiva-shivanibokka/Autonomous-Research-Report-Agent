"""
Reproductions of two review findings, written before the fixes.

1. The Tavily credit cap raised an ordinary Exception, which the Search and
   Fact-Checker agents' broad `except Exception` turned into an empty result,
   so the run continued as if nothing happened.
2. run_one() recorded a run as successful even when the pipeline set
   fatal_error / errors.
"""

import asyncio

from eval_sop import common


def test_tavily_cap_is_not_swallowed_by_search_agent(monkeypatch):
    common.install()
    import agents.search_agent as search_agent
    import agents.tools.search_tool as search_tool  # noqa: F401

    async def never_called(*a, **k):  # the cap must stop us before this
        raise AssertionError("search should not be issued past the cap")

    monkeypatch.setattr(common, "TAVILY_CREDIT_CAP", 0)
    common._tavily_cache.db.execute("DELETE FROM kv")
    raised = None
    try:
        out = asyncio.run(search_agent.run_search_agent("a fresh query", job_id="t"))
    except BaseException as e:  # noqa: BLE001
        raised = e
        out = None
    assert raised is not None, f"cap was swallowed; agent returned error={out.error!r}"


def _fake_pipeline_env(monkeypatch, fail_agent=None):
    """Fake LLM/search/scrape; `fail_agent` raises a non-retryable error."""
    import json as _json

    import agents.fact_checker_agent as fca
    import agents.search_agent as sa
    import agents.tools.scraper_tool as st
    from agents.schemas import ScrapedPage, SearchResult

    patched = common.install()

    async def fake_chat(model, messages, *, max_tokens, agent=None, **kw):
        agent = agent or common.CURRENT_AGENT.get()
        if agent == fail_agent:
            raise RuntimeError(f"injected failure in {agent}")
        if agent == "orchestrator":
            body = {"sub_questions": ["sub q"], "reasoning": "r"}
        elif agent == "analyst":
            body = {"claims": [{"text": "c", "source_urls": ["https://a.example/1"],
                                "supporting_sources": 1, "contradicting_sources": 0}],
                    "contradictions_found": 0, "avg_confidence_score": 0.3}
        elif agent == "critic":
            body = {"flagged_claims": [], "overall_quality_score": 0.9, "coverage_score": 0.9,
                    "source_diversity_score": 0.9, "contradiction_rate": 0.0,
                    "needs_more_research": False, "critic_notes": "n"}
        elif agent == "writer":
            body = {"title": "T", "executive_summary": "S", "key_findings": [],
                    "detailed_sections": {}, "confidence_assessment": "c", "limitations": "l"}
        else:
            return "Final answer: x", 10, 5, False
        return _json.dumps(body), 10, 5, False

    async def fake_search(query, *, max_results=8, **_):
        return [SearchResult(url="https://a.example/1", title="t", snippet="s",
                             relevance_score=0.9, source_domain="a.example")]

    async def fake_scrape(url, title=""):
        return ScrapedPage(url=url, title="t", content="text", word_count=1)

    monkeypatch.setattr(common, "chat", fake_chat)
    monkeypatch.setattr(sa, "tavily_search", fake_search)
    monkeypatch.setattr(fca, "tavily_search", fake_search)
    monkeypatch.setattr(st, "scrape_page", fake_scrape)
    return patched


def test_run_with_fatal_pipeline_error_is_marked_errored(monkeypatch):
    from eval_sop import run_conditions as rc

    patched = _fake_pipeline_env(monkeypatch, fail_agent="writer")
    q = {"qid": "t1", "question": "What is the test question here?", "reference": "x"}
    rec = asyncio.run(rc.run_one("d", q, 1, patched))
    assert rec["pipeline"]["fatal_error"], "precondition: writer failure is fatal"
    assert rec["error"], "a run whose pipeline set fatal_error was recorded as a success"
