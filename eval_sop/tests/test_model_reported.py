"""Round-3 review: the model-mismatch check must use the model the API reports."""

import asyncio

from eval_sop import common
from eval_sop.tests.test_anthropic_budget import FakeAnthropic, _router, fake_web, paid  # noqa: F401


def test_reported_model_mismatch_marks_the_run_errored(paid, fake_web):  # noqa: F811
    from eval_sop import run_conditions as rc

    def sneaky(kw):
        m = _router(kw)
        m.model = "claude-sonnet-4-5-20250929"  # the API served a different model
        return m

    common.set_anthropic_client(FakeAnthropic(router=sneaky))
    rec = asyncio.run(rc.run_one("a", rc.CANARY, 0, fake_web))
    assert rec["error_kind"] == "model_mismatch", rec["error"]
