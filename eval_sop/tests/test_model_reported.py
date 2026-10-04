"""The model-mismatch check uses the model the API reports, and stops the run."""

import asyncio

import pytest

from eval_sop import budget, common
from eval_sop.tests.test_anthropic_budget import (  # noqa: F401
    MODEL,
    FakeAnthropic,
    _router,
    fake_web,
    paid,
)


def test_reported_model_mismatch_stops_the_run(paid, fake_web):  # noqa: F811
    """A run whose provider serves another model must not be saved as a result."""
    from eval_sop import run_conditions as rc

    def sneaky(kw):
        m = _router(kw)
        m.model = "claude-sonnet-4-5-20250929"  # the API served a different model
        return m

    common.set_anthropic_client(FakeAnthropic(router=sneaky))
    with pytest.raises(budget.BudgetStop, match="model mismatch"):
        asyncio.run(rc.run_one("a", rc.CANARY, 0, fake_web))


# --- round-4: a substituted model must stop the run at the transport ----------
SUBSTITUTE = "claude-opus-4-8"  # $5/$25 vs Haiku $1/$5


def _sub_client(usage_in=1000, usage_out=1000):
    """Fake whose response reports a different, pricier model than requested."""
    from eval_sop.tests.test_review2_ledger import Fake, _msg

    def router(i):
        m = _msg(inp=usage_in, out=usage_out)
        m.model = SUBSTITUTE
        return m

    return Fake(router)


def test_substituted_model_trips_the_stop_flag(paid):  # noqa: F811
    """A silently substituted model must be fatal at the transport, not flagged later."""
    fake = _sub_client()
    common.set_anthropic_client(fake)
    with pytest.raises(budget.BudgetStop):
        asyncio.run(common.chat(MODEL, [{"role": "user", "content": "hi"}],
                                max_tokens=1000, agent="t"))
    assert budget.STOP["reason"] and SUBSTITUTE in budget.STOP["reason"]
    # No second call may be made while the flag is set.
    with pytest.raises(budget.BudgetStop):
        asyncio.run(common.chat(MODEL, [{"role": "user", "content": "another"}],
                                max_tokens=1000, agent="t"))
    assert fake.calls == 1


def test_substituted_model_is_charged_at_worst_case_not_pinned_price(paid):  # noqa: F811
    common.set_anthropic_client(_sub_client())
    msgs = [{"role": "user", "content": "hi"}]
    with pytest.raises(budget.BudgetStop):
        asyncio.run(common.chat(MODEL, msgs, max_tokens=1000, agent="t"))
    rows = list(paid.db.execute("SELECT status, model, cost_usd FROM calls"))
    assert len(rows) == 1
    status, model_col, cost = rows[0]
    assert status == "model_mismatch" and model_col == SUBSTITUTE
    # Not the pinned-model price for the reported usage ($0.006): worst case.
    assert cost == pytest.approx(budget.worst_call_cost(MODEL, msgs, 1000))
