"""
Worst-case and expected cost of a run design, for --dry-run and for per-run
admission control. Nothing here calls an API.

WORST CASE is an upper bound per run built from the pipeline's own limits at
c0ae667 (every call's max_tokens, the scraper / writer truncation caps) and
these stated assumptions, which the code does not enforce:
  * at most 8 sub-questions per round (the Orchestrator is asked for 3-5, plus
    up to 3 Critic re-search queries in round 2);
  * at most 8 claims per sub-question (the Analyst is asked for 4-8);
  * Tavily snippets <= 1,500 chars.
Input characters are converted at 2.5 chars/token (Claude averages ~3.5-4 on
English), and every call is assumed to use its full max_tokens. Even if an
assumption is broken at run time, the USD ledger still refuses any call whose
own worst case would cross the cap (budget.UsdLedger.precheck).

EXPECTED is a rough estimate: the repo's recorded showcase run (2 rounds,
78,591 tokens; tokens_by_agent from frontend/public/demo/run.json) repriced at
Haiku 4.5 rates with an ASSUMED 80/20 input/output split. It is for planning
only and is never reported as measured.
"""

from __future__ import annotations

import math

from eval_sop.budget import WORST_CHARS_PER_TOKEN, call_cost

MAX_SUBQ_PER_ROUND = 8
MAX_CLAIMS_PER_SUBQ = 8


def _tok(chars: int) -> int:
    return math.ceil(chars / WORST_CHARS_PER_TOKEN) + 50


# (label, n_calls, worst input tokens per call, max output tokens per call)
def _calls(cond: str, shared_round1: bool = False) -> list[tuple[str, int, int, int]]:
    orch = ("orchestrator", 1, _tok(6_000), 1024)
    analyst = ("analyst", MAX_SUBQ_PER_ROUND, _tok(15_000), 2048)
    fact = ("fact_checker", 5, _tok(7_000), 512)
    writer = ("writer", 1, _tok(38_000), 16_000)
    # Retry path: rewrite (same input, 24k cap) or repair (input = the <=16k-token
    # first draft). Take the larger input of the two.
    writer_retry = ("writer_retry", 1, max(_tok(38_000), 16_000 + 200), 24_000)
    extract = ("eval_answer_extraction", 1, _tok(24_600), 512)

    def critic(n_rounds_reviewed):
        claims = MAX_SUBQ_PER_ROUND * MAX_CLAIMS_PER_SUBQ * n_rounds_reviewed
        return ("critic", 1, _tok(2_000 + MAX_SUBQ_PER_ROUND * 200 + claims * 350), 2048)

    if cond == "a":
        return [("closed_book", 1, _tok(800), 1024)]
    if cond == "b":
        return [("search1", 1, _tok(8 * 1_750 + 800), 1024)]
    if cond == "c":
        first = [] if shared_round1 else [orch, analyst]
        return first + [critic(1), fact, writer, writer_retry, extract]
    if cond == "d":
        return [orch, analyst, critic(1), orch, analyst, critic(2), fact, writer,
                writer_retry, extract]
    raise ValueError(cond)


def worst_usd(cond: str, model: str, shared_round1: bool = False) -> float:
    return sum(n * call_cost(model, inp, out) for _, n, inp, out in _calls(cond, shared_round1))


def worst_tavily(cond: str, shared_round1: bool = False) -> int:
    return {"a": 0, "b": 1, "c": (0 if shared_round1 else MAX_SUBQ_PER_ROUND) + 5,
            "d": 2 * MAX_SUBQ_PER_ROUND + 5}[cond]


SHOWCASE_TOKENS_R2 = 78_591  # whole 2-round run, frontend/public/demo/run.json
SHOWCASE_R1_SHARE = 0.45  # round 1: 5 of 13 sub-questions + 1 of 2 critic calls (approx.)


def expected_usd(cond: str, model: str, shared_round1: bool = False) -> float:
    def price(tokens):
        return call_cost(model, int(tokens * 0.8), int(tokens * 0.2))

    extraction = call_cost(model, 6_000, 150)
    if cond == "a":
        return call_cost(model, 120, 400)
    if cond == "b":
        return call_cost(model, 2_500, 400)
    if cond == "d":
        return price(SHOWCASE_TOKENS_R2) + extraction
    if cond == "c":
        r1_full = price(SHOWCASE_TOKENS_R2 * SHOWCASE_R1_SHARE)
        r1_shared = price(9_571 / 2 + 11_189)  # one critic call + writer (tokens_by_agent)
        return (r1_shared if shared_round1 else r1_full) + extraction
    raise ValueError(cond)


def expected_tavily(cond: str, shared_round1: bool = False) -> int:
    return {"a": 0, "b": 1, "c": 2 if shared_round1 else 7, "d": 13}[cond]


def design_table(conds: list[str], n_questions: int, n_seeds: int, model: str) -> dict:
    rows = []
    for cond in conds:
        shared = cond == "c" and "d" in conds
        rows.append({
            "condition": cond,
            "runs": n_questions * n_seeds,
            "shares_round1_with_d": shared,
            "worst_usd_per_run": round(worst_usd(cond, model, shared), 4),
            "expected_usd_per_run": round(expected_usd(cond, model, shared), 4),
            "worst_tavily_per_run": worst_tavily(cond, shared),
            "expected_tavily_per_run": expected_tavily(cond, shared),
        })
    tot = lambda k: sum(r[k] * r["runs"] for r in rows)  # noqa: E731
    return {
        "model": model,
        "rows": rows,
        "worst_usd_total": round(tot("worst_usd_per_run"), 2),
        "expected_usd_total": round(tot("expected_usd_per_run"), 2),
        "worst_tavily_total": tot("worst_tavily_per_run"),
        "expected_tavily_total": tot("expected_tavily_per_run"),
    }
