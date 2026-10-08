"""Round-2 review nits: atomic ledger export; admission STOP exits non-zero."""

import asyncio

import pytest

from eval_sop import budget, common
from eval_sop.tests.test_anthropic_budget import FakeAnthropic, _args, paid  # noqa: F401


def test_export_jsonl_is_atomic(tmp_path, monkeypatch):
    led = budget.UsdLedger(tmp_path / "l.sqlite", 1.0)
    for i in range(3):
        _, row = led.reserve(model="claude-haiku-4-5-20251001", messages=[{"role": "user", "content": "x"}],
                             max_tokens=10, run="r", agent="a")
        led.settle(row, status="ok", cost_usd=0.001)
    out = tmp_path / "l.jsonl"
    out.write_text("PREVIOUS EXPORT\n", encoding="utf-8")
    calls = {"n": 0}
    real = budget.json.dumps

    def boom(obj):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(obj)

    monkeypatch.setattr(budget.json, "dumps", boom)
    with pytest.raises(OSError):
        led.export_jsonl(out)
    assert out.read_text(encoding="utf-8") == "PREVIOUS EXPORT\n"  # never half-written


def test_admission_stop_exits_non_zero(paid):  # noqa: F811
    from eval_sop import run_conditions as rc

    paid.cap = 0.05  # less than one (d) run's worst case
    common.set_anthropic_client(FakeAnthropic(router=lambda kw: pytest.fail("no call expected")))
    code = asyncio.run(rc.main_async(_args(dry_run=False, admission_control=True,
                                           conditions="d", n_frames=1)))
    assert code != 0
