"""The project's hard maximum is $8 (round-3 review): no cap above it is accepted."""

import asyncio

import pytest

from eval_sop import budget, common
from eval_sop.tests.test_anthropic_budget import _args


def test_ledger_refuses_a_cap_above_8(tmp_path):
    assert budget.PROJECT_HARD_MAX_USD == 8.0
    with pytest.raises(ValueError):
        budget.UsdLedger(tmp_path / "l.sqlite", 12.0)
    budget.UsdLedger(tmp_path / "ok.sqlite", 8.0)  # 8 itself is allowed


def test_env_cap_of_12_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("SOP_USD_CAP", "12")
    with pytest.raises(ValueError):
        budget.UsdLedger(tmp_path / "l.sqlite", budget.cap_from_env())


def test_cli_usd_cap_12_is_refused(monkeypatch):
    from eval_sop import run_conditions as rc

    monkeypatch.setattr(common, "LEDGER", None)
    code = asyncio.run(rc.main_async(_args(usd_cap=12.0, n_frames=1)))
    assert code == 2
