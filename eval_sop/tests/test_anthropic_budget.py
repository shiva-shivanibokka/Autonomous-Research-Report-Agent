"""
Anthropic transport + USD ledger + hard cap, against a fake Anthropic client.
No network and no key: the fake returns SDK-shaped messages with `usage`, and
raises the SDK's own exception classes for 429 / 400 / 402.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from eval_sop import budget, common

MODEL = "claude-haiku-4-5-20251001"
_ids = itertools.count()


def _msg(text, inp=1000, out=200, stop="end_turn"):
    return SimpleNamespace(
        id=f"msg_{next(_ids)}",
        model=MODEL,
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


def _status_error(cls, code, etype="error", headers=None):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(code, request=req, headers=headers or {},
                          json={"type": "error", "error": {"type": etype, "message": "x"}})
    return cls(f"{code} {etype}", response=resp, body={"type": "error", "error": {"type": etype}})


class _Stream:
    def __init__(self, outcome):
        self.outcome = outcome

    async def __aenter__(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self

    async def __aexit__(self, *a):
        return False

    async def get_final_message(self):
        return self.outcome


class FakeAnthropic:
    """`script` is a list of outcomes (message or exception) consumed in order;
    when it runs out, `router(kwargs)` produces the reply."""

    def __init__(self, script=None, router=None):
        self.script = list(script or [])
        self.router = router
        self.calls = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.script.pop(0) if self.script else self.router(kwargs)
        return _Stream(outcome)


@pytest.fixture
def paid(tmp_path, monkeypatch):
    """Anthropic backend with a fresh ledger + empty response cache per test."""
    common.install()  # patch transports (idempotent)
    monkeypatch.setattr(common, "BACKEND", "anthropic")
    monkeypatch.setattr(common, "GEN_MODEL", MODEL)
    monkeypatch.setattr(common, "LEDGER", budget.UsdLedger(tmp_path / "ledger.sqlite", 1.0))

    async def no_sleep(_):
        return None

    monkeypatch.setattr(common, "_sleep", no_sleep)
    common._llm_cache.db.execute("DELETE FROM kv")
    common._llm_cache.db.commit()
    yield common.LEDGER
    common.set_anthropic_client(None)


def _chat(text="hi", max_tokens=100, seed=0):
    return asyncio.run(common.chat(MODEL, [{"role": "system", "content": "s"},
                                           {"role": "user", "content": text}],
                                   max_tokens=max_tokens, seed=seed, agent="t"))


def test_usage_is_recorded_at_list_price(paid):
    fake = FakeAnthropic([_msg("ok", inp=1_000_000, out=100_000)])
    common.set_anthropic_client(fake)
    text, inp, out, cached = _chat()
    assert (text, inp, out, cached) == ("ok", 1_000_000, 100_000, False)
    assert paid.spent() == pytest.approx(1.00 + 0.50)  # $1/M in + $5/M out
    assert fake.calls[0]["model"] == MODEL and fake.calls[0]["system"] == "s"
    assert "temperature" not in fake.calls[0]


def test_response_cache_prevents_paying_twice(paid):
    fake = FakeAnthropic([_msg("first")])
    common.set_anthropic_client(fake)
    _chat()
    spent = paid.spent()
    text, *_, cached = _chat()  # same prompt, same replicate index
    assert cached and text == "first" and len(fake.calls) == 1 and paid.spent() == spent


def test_429_is_retried_then_succeeds(paid):
    fake = FakeAnthropic([_status_error(anthropic.RateLimitError, 429, "rate_limit_error",
                                        {"retry-after": "1"}), _msg("after 429")])
    common.set_anthropic_client(fake)
    assert _chat()[0] == "after 429"
    statuses = [r[0] for r in paid.db.execute("SELECT status FROM calls ORDER BY id")]
    assert statuses == ["429", "ok"]


def test_429_gives_up_after_two_retries(paid):
    err = _status_error(anthropic.RateLimitError, 429, "rate_limit_error")
    fake = FakeAnthropic([err, err, err, _msg("never")])
    common.set_anthropic_client(fake)
    with pytest.raises(common.TransportError, match="gave up after 2 retries"):
        _chat()
    assert len(fake.calls) == 3


def test_400_fails_fast_without_retry(paid):
    fake = FakeAnthropic([_status_error(anthropic.BadRequestError, 400, "invalid_request_error"),
                          _msg("never")])
    common.set_anthropic_client(fake)
    with pytest.raises(common.TransportError, match="400"):
        _chat()
    assert len(fake.calls) == 1


def test_billing_error_stops_everything(paid):
    fake = FakeAnthropic([_status_error(anthropic.APIStatusError, 402, "billing_error"),
                          _msg("never")])
    common.set_anthropic_client(fake)
    with pytest.raises(budget.BudgetStop):
        _chat()
    with pytest.raises(budget.BudgetStop):  # flag stays tripped for every later call
        _chat("another prompt")
    assert len(fake.calls) == 1


def test_cap_refuses_call_whose_worst_case_does_not_fit(paid):
    fake = FakeAnthropic(router=lambda kw: _msg("ok", inp=10, out=10))
    common.set_anthropic_client(fake)
    paid.cap = 0.01
    with pytest.raises(budget.BudgetStop, match="USD cap"):
        _chat(max_tokens=10_000)  # worst case 10k x $5/M = $0.05 > $0.01
    assert fake.calls == []  # refused before any request was sent
    assert budget.STOP["reason"]


def test_model_outside_allow_list_is_refused(paid):
    with pytest.raises(ValueError, match="allow list"):
        asyncio.run(common.chat("claude-opus-5-5", [{"role": "user", "content": "x"}],
                                max_tokens=10, agent="t"))


def test_concurrent_calls_reserve_their_worst_case(paid):
    """Two parallel calls whose worst cases fit only one at a time: one is refused."""
    gate = asyncio.Event()

    class SlowStream(_Stream):
        async def get_final_message(self):
            await gate.wait()
            return _msg("ok", inp=10, out=10)

    class Slow(FakeAnthropic):
        def _stream(self, **kw):
            self.calls.append(kw)
            return SlowStream(None)

    fake = Slow()
    common.set_anthropic_client(fake)
    paid.cap = 0.008  # one call's worst case (~$0.005) fits, two do not

    async def both():
        t1 = asyncio.create_task(common.chat(MODEL, [{"role": "user", "content": "a"}],
                                             max_tokens=1000, agent="t"))
        await asyncio.sleep(0)
        t2 = asyncio.create_task(common.chat(MODEL, [{"role": "user", "content": "b"}],
                                             max_tokens=1000, agent="t"))
        await asyncio.sleep(0.01)
        gate.set()
        return await asyncio.gather(t1, t2, return_exceptions=True)

    r1, r2 = asyncio.run(both())
    assert isinstance(r2, budget.BudgetStop) and not isinstance(r1, BaseException)
    assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# Whole runs through run_conditions with the fake client
# ---------------------------------------------------------------------------
def _router(kw):
    system = kw.get("system", "")
    user = kw["messages"][-1]["content"]
    if "decomposition expert" in system:
        body = {"sub_questions": ["sub q one", "sub q two"], "reasoning": "r"}
    elif "extract key claims" in system:
        body = {"claims": [{"text": f"claim about {user[23:35]}", "source_urls": ["https://a.example/1"],
                            "supporting_sources": 2, "contradicting_sources": 0}],
                "contradictions_found": 0, "avg_confidence_score": 0.6}
    elif "research critic" in system:
        first = "Research round: 1 of" in user
        body = {"flagged_claims": [{"claim_text": "claim about sub q one", "reason": "r",
                                    "re_search_query": "verify"}] if first else [],
                "overall_quality_score": 0.6, "coverage_score": 0.6, "source_diversity_score": 0.5,
                "contradiction_rate": 0.1, "needs_more_research": first, "critic_notes": "n"}
    elif "fact-checker" in system:
        body = {"verdict": "verified", "evidence": "e", "confidence": "medium", "explanation": "x"}
    elif "comprehensive report" in system:
        body = {"title": "T", "executive_summary": "The capital is Paris.", "key_findings": ["k"],
                "detailed_sections": {"A": "a"}, "confidence_assessment": "c", "limitations": "l"}
    else:
        return _msg("Reasoning.\nFinal answer: Paris", inp=300, out=20)
    return _msg(json.dumps(body), inp=2000, out=300)


@pytest.fixture
def fake_web(monkeypatch):
    import agents.fact_checker_agent as fca
    import agents.search_agent as sa
    import agents.tools.scraper_tool as st
    from agents.schemas import ScrapedPage, SearchResult

    async def fake_search(query, *, max_results=8, **_):
        return [SearchResult(url="https://a.example/1", title="t", snippet="Paris is the capital.",
                             relevance_score=0.9, source_domain="a.example")]

    async def fake_scrape(url, title=""):
        return ScrapedPage(url=url, title="t", content="Paris is the capital of France.", word_count=6)

    monkeypatch.setattr(sa, "tavily_search", fake_search)
    monkeypatch.setattr(fca, "tavily_search", fake_search)
    monkeypatch.setattr(st, "scrape_page", fake_scrape)
    return SimpleNamespace(search=fake_search, scrape=fake_scrape)


def test_pipeline_run_uses_only_the_pinned_model_and_is_costed(paid, fake_web):
    from eval_sop import run_conditions as rc

    fake = FakeAnthropic(router=_router)
    common.set_anthropic_client(fake)
    rec = asyncio.run(rc.run_one("d", rc.CANARY, 0, fake_web))
    assert rec["error"] is None, rec["error"]
    assert {c["model"] for c in fake.calls} == {MODEL}
    assert rec["telemetry"]["models"] == [MODEL]
    assert rec["pipeline"]["rounds_run"] == 2
    assert rec["telemetry"]["usd_spent_now"] == pytest.approx(paid.spent())
    assert "paris" in rec["final_answer"].lower()


def test_c_after_d_replays_round_one_from_cache(paid, fake_web):
    from eval_sop import run_conditions as rc

    fake = FakeAnthropic(router=_router)
    common.set_anthropic_client(fake)
    asyncio.run(rc.run_one("d", rc.CANARY, 0, fake_web))
    rec_c = asyncio.run(rc.run_one("c", rc.CANARY, 0, fake_web))
    tel = rec_c["telemetry"]
    assert tel["llm_calls_from_cache"] >= 1 + 2  # orchestrator + 2 analysts replayed
    assert tel["usd_spent_now"] < tel["usd_if_uncached"]


def test_cap_hit_mid_pipeline_stops_the_run_and_ledger_survives(paid, fake_web, tmp_path):
    from eval_sop import run_conditions as rc

    fake = FakeAnthropic(router=_router)
    common.set_anthropic_client(fake)
    paid.cap = 0.03  # enough for a few calls, not for the Writer's 16k-token worst case
    with pytest.raises(budget.BudgetStop):
        asyncio.run(rc.run_one("d", rc.CANARY, 0, fake_web))
    n_calls = len(fake.calls)
    spent = paid.spent()
    assert 0 < spent <= paid.cap and n_calls > 0
    # The ledger is on disk: a new process (new connection) sees the same spend.
    reopened = budget.UsdLedger(paid.path, paid.cap)
    assert reopened.spent() == pytest.approx(spent)


def test_crash_mid_run_then_resume_does_not_pay_twice(paid, fake_web):
    from eval_sop import run_conditions as rc

    calls = {"n": 0}

    def crashing_router(kw):
        calls["n"] += 1
        if "comprehensive report" in kw.get("system", "") and calls.setdefault("crashed", 0) == 0:
            calls["crashed"] = 1
            raise KeyboardInterrupt("simulated crash")  # escapes everything, like a kill
        return _router(kw)

    common.set_anthropic_client(FakeAnthropic(router=crashing_router))
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(rc.run_one("d", rc.CANARY, 0, fake_web))
    spent_before = paid.spent()
    paid_before = paid.db.execute("SELECT COUNT(*) FROM calls WHERE status='ok'").fetchone()[0]
    fake2 = FakeAnthropic(router=_router)
    common.set_anthropic_client(fake2)
    rec = asyncio.run(rc.run_one("d", rc.CANARY, 0, fake_web))
    assert rec["error"] is None
    # Everything before the Writer replayed from cache; only the rest was paid.
    assert all("comprehensive report" in c.get("system", "") or "answer questions using only" in
               c.get("system", "") for c in fake2.calls), [c.get("system", "")[:40] for c in fake2.calls]
    paid_after = paid.db.execute("SELECT COUNT(*) FROM calls WHERE status='ok'").fetchone()[0]
    assert paid_after - paid_before == len(fake2.calls)
    assert paid.spent() > spent_before


def test_pipeline_errors_mark_the_run_errored(paid, fake_web):
    from eval_sop import run_conditions as rc

    def bad_writer(kw):
        if "comprehensive report" in kw.get("system", ""):
            return _status_error(anthropic.BadRequestError, 400, "invalid_request_error")
        return _router(kw)

    common.set_anthropic_client(FakeAnthropic(router=bad_writer))
    rec = asyncio.run(rc.run_one("d", rc.CANARY, 0, fake_web))
    assert rec["error_kind"] == "fatal_error" and "Writer failed" in rec["error"]


def _args(**kw):
    base = dict(backend="anthropic", env_file=None, conditions="a,b,d", seeds="1", n_frames=30,
                open=False, only="", force=False, usd_cap=8.0, tavily_cap=600, dry_run=True,
                canary=False, admission_control=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_dry_run_refuses_a_design_whose_worst_case_exceeds_the_cap(paid, capsys):
    from eval_sop import run_conditions as rc

    paid.cap = 8.0
    assert asyncio.run(rc.main_async(_args())) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert asyncio.run(rc.main_async(_args(n_frames=12))) == 0  # worst case fits $8


def test_real_run_without_admission_control_is_refused_too(paid):
    from eval_sop import run_conditions as rc

    paid.cap = 8.0
    common.set_anthropic_client(FakeAnthropic(router=lambda kw: pytest.fail("no call expected")))
    assert asyncio.run(rc.main_async(_args(dry_run=False))) == 2
