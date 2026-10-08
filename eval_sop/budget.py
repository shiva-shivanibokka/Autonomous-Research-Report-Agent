"""
Hard spending caps for the evaluation: a persisted USD ledger for paid LLM
calls and a stop signal the pipeline's agents cannot swallow.

Why a BaseException: every agent wraps its LLM / search call in
`except Exception` and turns failures into empty results (by design — one
failed branch should not kill a report). A cap that raised an ordinary
Exception was therefore silently converted into "no results" and the run
carried on. BudgetStop derives from BaseException so those handlers do not
catch it, and it also sets a process-wide flag (`STOP`) because an exception
raised inside `asyncio.gather(..., return_exceptions=True)` is handed back as
a value rather than raised. Every transport call checks the flag first, and
run_conditions.run_one checks it after each run, so once tripped nothing else
is spent and the run in progress is not saved as a result.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import sqlite3
import time
from pathlib import Path

# USD per 1M tokens, Anthropic first-party list prices (checked 2026-10-04).
# Only models in this table may be called; anything else is refused, which is
# also how thinking-capable 5.x models are kept out (no thinking handling here).
PRICES = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
}
# Hard ceiling for this project regardless of what --usd-cap says.
PROJECT_HARD_MAX_USD = 8.0
# Worst-case input estimate: 2.5 chars/token over-counts English for Claude's
# tokenizer (~3.5-4 chars/token), so the pre-call check errs on the safe side.
WORST_CHARS_PER_TOKEN = 2.5

STOP: dict[str, str | None] = {"reason": None}


class BudgetStop(BaseException):  # noqa: N818 - it is a stop signal, not an error
    """A hard cap would be exceeded, or the provider reported a billing problem."""


def trip(reason: str) -> None:
    STOP["reason"] = reason
    raise BudgetStop(reason)


def check_stop() -> None:
    if STOP["reason"]:
        raise BudgetStop(STOP["reason"])


def reset_stop() -> None:  # tests only
    STOP["reason"] = None


def _text_of(content) -> str:
    """Message content as text: a string, or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(getattr(b, "text", b)) for b in content
        )
    return str(content)


def worst_input_tokens(messages: list[dict]) -> int:
    """max(chars/2.5, UTF-8 bytes/3): the byte term keeps CJK and other
    multi-byte text (roughly one token per character or more) from being
    undercounted by the character term, which is tuned for English."""
    texts = [_text_of(m.get("content")) for m in messages]
    chars = sum(len(t) for t in texts)
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    return math.ceil(max(chars / WORST_CHARS_PER_TOKEN, nbytes / 3)) + 50


def price_of(model: str) -> dict:
    if model not in PRICES:
        raise ValueError(f"model {model!r} is not in the eval's price/allow list")
    return PRICES[model]


def call_cost(model: str, inp: int, out: int, cache_write: int = 0, cache_read: int = 0) -> float:
    p = price_of(model)
    return (
        inp * p["input"] + out * p["output"] + cache_write * p["cache_write"] + cache_read * p["cache_read"]
    ) / 1e6


def worst_call_cost(model: str, messages: list[dict], max_tokens: int) -> float:
    return call_cost(model, worst_input_tokens(messages), max_tokens)


def default_state_dir() -> Path:
    """
    Per-user, per-project state OUTSIDE any checkout: the USD ledger, the Tavily
    credit ledger and the run lock. A run from the worktree and one from the
    main checkout therefore share one ledger, and no output-directory setting
    can point the spend at a fresh file. Tests monkeypatch common.STATE_DIR.

    `Path.home()` on every platform, never `%LOCALAPPDATA%`: that variable is
    routinely set per-process, and moving it moved both the ledger — a fresh $0
    total, so the canary's spend was forgotten and the next run got the whole
    cap again — and the run lock beside it, letting two paid runs overlap.
    There is deliberately no override variable of our own. One caveat, stated
    rather than glossed: `Path.home()` on Windows reads `USERPROFILE`, so that
    one variable does still move this path. It is not the hole `LOCALAPPDATA`
    was -- the OS sets `USERPROFILE` at logon and redirecting it breaks the whole
    session, whereas tools set `LOCALAPPDATA` per process as a matter of course.
    Verified: `LOCALAPPDATA`, `APPDATA`, `XDG_STATE_HOME`, `HOME`, `HOMEDRIVE`,
    `HOMEPATH`, `TEMP` and `TMP` all leave it unmoved, individually and together.
    """
    return Path.home() / ".sop_eval" / "research_report"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None: we issue BEGIN IMMEDIATE / COMMIT ourselves, so a
    # check-then-write is one transaction even across processes; timeout makes
    # a second writer wait for the lock instead of failing.
    return sqlite3.connect(path, timeout=60, isolation_level=None, check_same_thread=False)


class UsdLedger:
    """
    sqlite ledger of every paid request, written BEFORE the request is sent.

    reserve() checks the cap and inserts the attempt's row ('pending', cost =
    its worst case) in ONE `BEGIN IMMEDIATE` transaction, so two processes (or
    two connections) can never both pass the check on the same headroom; the
    row is committed, hence visible to every other process, before the request
    goes out. Only a successful response UPDATEs the row to the actual cost; an
    error the provider guarantees is unbilled (HTTP-level 429 / 5xx / 4xx
    rejection) UPDATEs it to 0. Every other way out — a mid-stream error, a
    timeout, a dropped connection, KeyboardInterrupt, task cancellation, a
    crash — leaves the worst-case charge in place.
    """

    def __init__(self, path: Path, cap_usd: float):
        check_cap(cap_usd)
        self.cap = cap_usd
        self.path = path
        self.db = _connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, ts TEXT, run TEXT, agent TEXT,"
            " model TEXT, status TEXT, input_tokens INT, output_tokens INT, cache_write INT,"
            " cache_read INT, cost_usd REAL, request_id TEXT, note TEXT)"
        )

    def spent(self) -> float:
        return float(self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()[0])

    def remaining(self) -> float:
        return self.cap - self.spent()

    def reserve(self, *, model: str, messages: list[dict], max_tokens: int, run: str,
                agent: str) -> tuple[float, int]:
        """Atomically: refuse (trip) if the worst case would cross the cap, else
        insert the pending worst-case row. Returns (worst, row_id)."""
        check_stop()
        worst = worst_call_cost(model, messages, max_tokens)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            spent = float(self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()[0])
            if spent + worst > self.cap:
                self.db.execute("ROLLBACK")
                trip(
                    f"USD cap ${self.cap:.2f} would be exceeded: spent incl. pending "
                    f"${spent:.4f} + worst case ${worst:.4f} for the next call"
                )
            cur = self.db.execute(
                "INSERT INTO calls (ts, run, agent, model, status, input_tokens, output_tokens,"
                " cache_write, cache_read, cost_usd, request_id, note)"
                " VALUES (?,?,?,?, 'pending', 0, 0, 0, 0, ?, NULL, 'charged at worst case until settled')",
                (time.strftime("%Y-%m-%dT%H:%M:%S"), run, agent, model, worst),
            )
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        return worst, cur.lastrowid

    def settle(self, row_id: int, *, status: str, cost_usd: float, input_tokens: int = 0,
               output_tokens: int = 0, cache_write: int = 0, cache_read: int = 0,
               request_id: str | None = None, note: str = "") -> None:
        self.db.execute(
            "UPDATE calls SET status=?, cost_usd=?, input_tokens=?, output_tokens=?, cache_write=?,"
            " cache_read=?, request_id=?, note=? WHERE id=?",
            (status, cost_usd, input_tokens, output_tokens, cache_write, cache_read, request_id,
             note, row_id),
        )

    def export_jsonl(self, out: Path) -> None:
        """Write to a temp file, then os.replace: the export is never half-written."""
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = [d[0] for d in self.db.execute("SELECT * FROM calls LIMIT 0").description]
        tmp = out.with_name(out.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for row in self.db.execute("SELECT * FROM calls ORDER BY id"):
                    f.write(json.dumps(dict(zip(cols, row))) + "\n")
            os.replace(tmp, out)
        finally:
            if tmp.exists():
                tmp.unlink()


class CreditLedger:
    """Tavily credits booked by this project, incremented atomically across processes."""

    def __init__(self, path: Path):
        self.path = path
        self.db = _connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS credits (k TEXT PRIMARY KEY, v INTEGER NOT NULL)")
        self.db.execute("INSERT OR IGNORE INTO credits (k, v) VALUES ('spent', 0)")

    def spent(self) -> int:
        return int(self.db.execute("SELECT v FROM credits WHERE k='spent'").fetchone()[0])

    def reserve(self, n: int, cap: int) -> bool:
        """Book n credits if spent + n <= cap (one BEGIN IMMEDIATE transaction)."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            spent = int(self.db.execute("SELECT v FROM credits WHERE k='spent'").fetchone()[0])
            if spent + n > cap:
                self.db.execute("ROLLBACK")
                return False
            self.db.execute("UPDATE credits SET v = v + ? WHERE k='spent'", (n,))
            self.db.execute("COMMIT")
            return True
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise


class EvalLocked(RuntimeError):
    """Another evaluation process holds this project's run lock."""


class ProcessLock:
    """
    One evaluation process per project at a time: an O_CREAT|O_EXCL lock file
    in the state directory, held for the life of the process (re-entrant within
    it) and removed at normal exit — including after Ctrl-C, which unwinds as a
    KeyboardInterrupt through atexit. A hard kill leaves it behind; the error
    message then says how to recover.
    """

    def __init__(self, path: Path):
        self.path = path
        self.held = False

    def acquire(self) -> None:
        if self.held:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        info = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                other = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                other = {}
            if other.get("pid") == os.getpid():
                self.held = True
                return
            raise EvalLocked(
                f"another evaluation process holds {self.path} (pid {other.get('pid')}, "
                f"started {other.get('started')}). Only one may spend at a time. If that "
                f"process is no longer running the lock is stale: check with "
                f"`tasklist /FI \"PID eq {other.get('pid')}\"`, then delete the file and rerun."
            ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(info, f)
        self.held = True
        atexit.register(self.release)

    def release(self) -> None:
        if not self.held:
            return
        try:
            if json.loads(self.path.read_text(encoding="utf-8")).get("pid") == os.getpid():
                self.path.unlink()
        except (OSError, ValueError):
            pass
        self.held = False


def check_cap(cap_usd) -> float:
    """
    A cap must be a finite, positive number no greater than the project max.
    `nan` is the reason this is a function: it fails EVERY comparison, so a
    bare `cap > MAX` guard let `--usd-cap nan` through, and every later
    `spent + worst > cap` was False too — a ledger with no cap at all.
    """
    # Accepts a string as well as a number: the CLI does NOT use this as
    # argparse's `type` (that is plain `float`, run_conditions.py:468) — the
    # parsed value is validated separately at run_conditions.py:373, and
    # `SOP_USD_CAP` arrives via cap_from_env() below.
    cap_usd = float(cap_usd)
    if not math.isfinite(cap_usd) or cap_usd <= 0:
        raise ValueError(f"cap {cap_usd!r} must be a finite positive number of dollars")
    if cap_usd > PROJECT_HARD_MAX_USD:
        raise ValueError(f"cap ${cap_usd} exceeds this project's hard max ${PROJECT_HARD_MAX_USD}")
    return cap_usd


def cap_from_env(default: float = 8.0) -> float:
    return check_cap(float(os.environ.get("SOP_USD_CAP", default)))
