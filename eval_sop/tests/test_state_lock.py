"""
Round-3 review: one ledger per project outside the checkout, one spending
process at a time, and cap checks that are atomic across processes.
Scenarios from scratchpad/review3/rr/{race.py,attack3.py}.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from eval_sop import budget, common

WT = Path(__file__).resolve().parents[2]
MODEL = "claude-haiku-4-5-20251001"
M = [{"role": "user", "content": "hi"}]


def test_state_lives_outside_the_checkout_and_out_dir_cannot_move_it(monkeypatch, tmp_path):
    d = budget.default_state_dir()
    assert d.parts[-2:] == (".sop_eval", "research_report")
    assert WT not in d.parents
    monkeypatch.setenv("SOP_LEDGER_PATH", str(tmp_path / "elsewhere.sqlite"))  # no longer honoured
    monkeypatch.setattr(common, "OUT_DIR", tmp_path / "fresh")
    assert common.ledger_path() == common.STATE_DIR / "usd_ledger.sqlite"
    assert common.tavily_ledger_path().parent == common.lock_path().parent == common.STATE_DIR


def test_state_dir_is_under_the_home_directory():
    """Round-5: the path is derived from Path.home(), so no variable can move it."""
    d = budget.default_state_dir()
    assert Path.home() in d.parents
    assert d.parts[-2:] == (".sop_eval", "research_report")


_RESOLVE = (
    "import sys, json, importlib;"
    f" sys.path.insert(0, r'{WT}');"
    " import eval_sop.budget as b, eval_sop.common as c;"
    " importlib.reload(b); importlib.reload(c);"
    " print(json.dumps([str(b.default_state_dir()), str(c.ledger_path()),"
    " str(c.lock_path()), str(c.tavily_ledger_path())]))"
)


def _resolve_in_child(env=None):
    e = dict(os.environ, TAVILY_API_KEY="x")
    e.update(env or {})
    r = subprocess.run([sys.executable, "-c", _RESOLVE], capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_state_dir_ignores_localappdata_in_a_fresh_process(tmp_path):
    """
    The gate failure: with LOCALAPPDATA redirected, a dry run reported an empty
    ledger — a fresh cap's worth of headroom, the canary's spend forgotten —
    and the run lock moved with it, so two paid runs could overlap. Resolved in
    a child process (and with both modules reloaded there), because an import-
    time constant is exactly what a new shell re-evaluates.
    """
    redirected = tmp_path / "redirected"
    before = _resolve_in_child()
    after = _resolve_in_child({"LOCALAPPDATA": str(redirected)})
    assert after == before
    assert not any(str(redirected) in p for p in after)
    assert all(str(Path.home()) in p for p in after)


def test_lock_sits_beside_the_ledger(monkeypatch):
    """A lock that can move without the ledger lets two runs share one cap."""
    monkeypatch.setattr(common, "STATE_DIR", budget.default_state_dir())
    assert common.lock_path().parent == common.ledger_path().parent == budget.default_state_dir()
    assert common.tavily_ledger_path().parent == budget.default_state_dir()


def test_two_handles_cannot_both_pass_the_check(tmp_path):
    """attack3 #1: precheck/precheck/begin/begin overshot the cap; reserve() is atomic."""
    p = tmp_path / "l.sqlite"
    a, b = budget.UsdLedger(p, 0.006), budget.UsdLedger(p, 0.006)
    a.reserve(model=MODEL, messages=M, max_tokens=1000, run="r", agent="a")
    with pytest.raises(budget.BudgetStop):
        b.reserve(model=MODEL, messages=M, max_tokens=1000, run="r", agent="b")
    budget.reset_stop()
    assert a.spent() <= a.cap


def test_pending_row_is_visible_to_other_processes_before_the_request(tmp_path):
    """attack3 #3: the row is committed before the request goes out."""
    import sqlite3

    led = budget.UsdLedger(tmp_path / "l.sqlite", 1.0)
    led.reserve(model=MODEL, messages=M, max_tokens=100, run="r", agent="a")
    other = sqlite3.connect(tmp_path / "l.sqlite")
    assert other.execute("SELECT status FROM calls").fetchall() == [("pending",)]


def test_crash_after_reserve_is_still_charged_on_resume(tmp_path):
    """attack3 #2: os._exit right after the pending row was written."""
    p = tmp_path / "l.sqlite"
    code = (f"import sys; sys.path.insert(0, r'{WT}'); from eval_sop import budget; "
            f"from pathlib import Path; L = budget.UsdLedger(Path(r'{p}'), 1.0); "
            f"L.reserve(model='{MODEL}', messages=[{{'role':'user','content':'x'}}], max_tokens=1000,"
            f" run='r', agent='a'); import os; os._exit(9)")
    subprocess.run([sys.executable, "-c", code], check=False)
    assert budget.UsdLedger(p, 1.0).spent() > 0


# --- cross-process races (race.py), on the ledger primitives ------------------
def _usd_worker(path, cap, start, q):
    from eval_sop import budget as b

    led = b.UsdLedger(Path(path), cap)
    while time.time() < start:
        time.sleep(0.001)
    n = 0
    for j in range(10_000):
        try:
            _, row = led.reserve(model=MODEL, messages=[{"role": "user", "content": f"{os.getpid()} {j}"}],
                                 max_tokens=1000, run="r", agent="a")
        except b.BudgetStop:
            break
        led.settle(row, status="ok", cost_usd=0.004)  # below worst case, like a real call
        n += 1
    q.put(n)


def _credit_worker(path, cap, start, q):
    from eval_sop import budget as b

    led = b.CreditLedger(Path(path))
    while time.time() < start:
        time.sleep(0.001)
    n = 0
    while led.reserve(1, cap):
        n += 1
    q.put(n)


def _race(target, path, cap, nproc=4):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    start = time.time() + 4
    ps = [ctx.Process(target=target, args=(str(path), cap, start, q)) for _ in range(nproc)]
    for p in ps:
        p.start()
    got = [q.get(timeout=240) for _ in ps]
    for p in ps:
        p.join(timeout=60)
    return got


def test_four_processes_cannot_overshoot_the_usd_cap(tmp_path):
    p = tmp_path / "usd.sqlite"
    _race(_usd_worker, p, 0.5)
    assert budget.UsdLedger(p, 0.5).spent() <= 0.5 + 1e-9


def test_four_processes_cannot_overshoot_the_tavily_cap(tmp_path):
    p = tmp_path / "credits.sqlite"
    got = _race(_credit_worker, p, 200)
    assert sum(got) == 200 and budget.CreditLedger(p).spent() == 200


# --- the run lock --------------------------------------------------------------
def _child(state, body):
    return (f"import sys; sys.path.insert(0, r'{WT}'); import os; os.environ['TAVILY_API_KEY']='x';"
            f" from pathlib import Path; from eval_sop import common;"
            f" common.STATE_DIR = Path(r'{state}'); common.install(); {body}")


def test_second_process_is_refused_while_the_lock_is_held(tmp_path):
    state = tmp_path / "state"
    holder = subprocess.Popen([sys.executable, "-c", _child(state, "import time; print('held', flush=True); time.sleep(60)")],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        r = subprocess.run([sys.executable, "-c", _child(state, "print('started')")],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "EvalLocked" in r.stderr and "started" not in r.stdout
    finally:
        holder.kill()
        holder.wait()


@pytest.mark.parametrize("body", ["pass", "raise KeyboardInterrupt"])
def test_lock_is_released_on_normal_exit_and_ctrl_c(tmp_path, body):
    state = tmp_path / "state"
    subprocess.run([sys.executable, "-c", _child(state, body)], capture_output=True)
    assert not (state / "run.lock").exists()


def test_stale_lock_message_says_how_to_recover(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "run.lock").write_text(json.dumps({"pid": 999999, "started": "2026-01-01T00:00:00"}))
    lock = budget.ProcessLock(state / "run.lock")
    with pytest.raises(budget.EvalLocked, match="delete the file"):
        lock.acquire()
