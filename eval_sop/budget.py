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


def worst_input_tokens(messages: list[dict]) -> int:
    """max(chars/2.5, UTF-8 bytes/3): the byte term keeps CJK and other
    multi-byte text (roughly one token per character or more) from being
    undercounted by the character term, which is tuned for English."""
    texts = [m.get("content") or "" for m in messages]
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


class UsdLedger:
    """
    sqlite ledger of every paid request, written BEFORE the request is sent.

    Each attempt first inserts a row with status 'pending' and cost = its worst
    case. Only a successful response UPDATEs the row to the actual cost; an
    error the provider guarantees is unbilled (HTTP-level 429 / 5xx / 4xx
    rejection) UPDATEs it to 0. Every other way out — a mid-stream error, a
    timeout, a dropped connection, KeyboardInterrupt, task cancellation, a
    crash — leaves the worst-case charge in place. Because pending rows count
    in spent(), parallel calls also see each other's worst cases.
    """

    def __init__(self, path: Path, cap_usd: float):
        if cap_usd > PROJECT_HARD_MAX_USD:
            raise ValueError(f"cap ${cap_usd} exceeds this project's hard max ${PROJECT_HARD_MAX_USD}")
        self.cap = cap_usd
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, ts TEXT, run TEXT, agent TEXT,"
            " model TEXT, status TEXT, input_tokens INT, output_tokens INT, cache_write INT,"
            " cache_read INT, cost_usd REAL, request_id TEXT, note TEXT)"
        )
        self.db.commit()

    def spent(self) -> float:
        return float(self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()[0])

    def remaining(self) -> float:
        return self.cap - self.spent()

    def precheck(self, model: str, messages: list[dict], max_tokens: int) -> float:
        """Refuse (and trip the stop flag) if the call's worst case could cross the cap."""
        check_stop()
        worst = worst_call_cost(model, messages, max_tokens)
        if self.spent() + worst > self.cap:
            trip(
                f"USD cap ${self.cap:.2f} would be exceeded: spent incl. pending "
                f"${self.spent():.4f} + worst case ${worst:.4f} for the next call"
            )
        return worst

    def begin(self, *, run: str, agent: str, model: str, worst: float) -> int:
        """Insert the pending worst-case row for one request attempt; returns its id."""
        cur = self.db.execute(
            "INSERT INTO calls (ts, run, agent, model, status, input_tokens, output_tokens,"
            " cache_write, cache_read, cost_usd, request_id, note)"
            " VALUES (?,?,?,?, 'pending', 0, 0, 0, 0, ?, NULL, 'charged at worst case until settled')",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), run, agent, model, worst),
        )
        self.db.commit()
        return cur.lastrowid

    def settle(self, row_id: int, *, status: str, cost_usd: float, input_tokens: int = 0,
               output_tokens: int = 0, cache_write: int = 0, cache_read: int = 0,
               request_id: str | None = None, note: str = "") -> None:
        self.db.execute(
            "UPDATE calls SET status=?, cost_usd=?, input_tokens=?, output_tokens=?, cache_write=?,"
            " cache_read=?, request_id=?, note=? WHERE id=?",
            (status, cost_usd, input_tokens, output_tokens, cache_write, cache_read, request_id,
             note, row_id),
        )
        self.db.commit()

    def record(self, *, run: str, agent: str, model: str, status: str, input_tokens: int = 0,
               output_tokens: int = 0, cache_write: int = 0, cache_read: int = 0,
               cost_usd: float, request_id: str | None = None, note: str = "") -> None:
        self.db.execute(
            "INSERT INTO calls (ts, run, agent, model, status, input_tokens, output_tokens, cache_write,"
            " cache_read, cost_usd, request_id, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), run, agent, model, status, input_tokens,
             output_tokens, cache_write, cache_read, cost_usd, request_id, note),
        )
        self.db.commit()

    def export_jsonl(self, out: Path) -> None:
        """Write to a temp file, then os.replace: the export is never half-written."""
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


def cap_from_env(default: float = 8.0) -> float:
    return float(os.environ.get("SOP_USD_CAP", default))
