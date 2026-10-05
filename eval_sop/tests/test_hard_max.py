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


@pytest.mark.parametrize("cap", ["nan", "inf", "-inf", "0", "-1"])
def test_non_finite_or_non_positive_caps_are_refused(tmp_path, cap, monkeypatch):
    """nan passed every `cap > 8` / `spent + worst > cap` comparison, disabling the cap."""
    with pytest.raises(ValueError):
        budget.UsdLedger(tmp_path / f"l-{cap}.sqlite", float(cap))
    monkeypatch.setenv("SOP_USD_CAP", cap)
    with pytest.raises(ValueError):
        budget.cap_from_env()


def test_cli_nan_cap_is_refused(monkeypatch):
    from eval_sop import run_conditions as rc

    monkeypatch.setattr(common, "LEDGER", None)
    assert asyncio.run(rc.main_async(_args(usd_cap=float("nan"), n_frames=1))) == 2


def test_corrupt_ledger_exits_2_not_1(tmp_path, monkeypatch):
    """Exit 1 is documented as 'canary failed'; a ledger we cannot read is 'refused'."""
    from eval_sop import run_conditions as rc

    state = tmp_path / "state"
    state.mkdir()
    (state / "usd_ledger.sqlite").write_bytes(b"this is not a database" * 100)
    monkeypatch.setattr(common, "STATE_DIR", state)
    monkeypatch.setattr(common, "LEDGER", None)
    assert asyncio.run(rc.main_async(_args(n_frames=1))) == 2


# --------------------------------------------------------------- round-5: the credit cap needs the same hard max
#
# The USD cap is hardened at four layers. The Tavily credit cap had none: it was
# `int(os.environ.get("SOP_TAVILY_CAP", "600"))`, so the environment could raise
# it without limit, and `--tavily-cap` took any int argparse would parse --
# including negative values, which make every headroom comparison nonsense.
# Credits are a paid resource on a shared key, so an unbounded cap is a real
# spend risk, not a tidiness issue.
def test_credit_cap_hard_max_exists_and_600_is_the_default():
    assert common.PROJECT_HARD_MAX_CREDITS == 600
    assert common.TAVILY_CREDIT_CAP <= common.PROJECT_HARD_MAX_CREDITS


@pytest.mark.parametrize("bad", ["601", "999999", "0", "-5", "-1", "nan", "inf", "", "abc", "1e9"])
def test_env_credit_cap_above_the_max_or_not_a_positive_int_is_refused(bad, monkeypatch):
    monkeypatch.setenv("SOP_TAVILY_CAP", bad)
    with pytest.raises(ValueError):
        common.credit_cap_from_env()


@pytest.mark.parametrize("good,expected", [("600", 600), ("1", 1), ("599", 599)])
def test_env_credit_cap_may_only_lower_it(good, expected, monkeypatch):
    monkeypatch.setenv("SOP_TAVILY_CAP", good)
    assert common.credit_cap_from_env() == expected


@pytest.mark.parametrize("bad", [601, 999999, 0, -5, float("nan"), float("inf")])
def test_check_credit_cap_refuses_bad_values(bad):
    with pytest.raises(ValueError):
        common.check_credit_cap(bad)


@pytest.mark.parametrize("bad", [601, 999999, 0, -5])
def test_cli_tavily_cap_above_the_max_is_refused_before_anything_installs(bad, monkeypatch):
    from eval_sop import run_conditions as rc

    monkeypatch.setattr(common, "LEDGER", None)
    code = asyncio.run(rc.main_async(_args(tavily_cap=bad, n_frames=1)))
    assert code == 2
