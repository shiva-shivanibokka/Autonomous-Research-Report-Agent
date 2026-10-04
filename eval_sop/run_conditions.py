"""
Run the four conditions on the question set and save one JSON per run.

  (a) closed_book : one LLM call, no search
  (b) search1     : one Tavily search on the question + one summarising LLM call
  (c) pipeline_r1 : the LangGraph pipeline, max_rounds=1
  (d) pipeline_r2 : the LangGraph pipeline, max_rounds=2

For every condition the final short answer is produced the same way the
condition naturally would: (a)/(b) are asked for a "Final answer:" line; for
(c)/(d) an extra, clearly separate extraction call (same generator model, same
seed) reads the rendered report and answers the question from it alone. That
extraction call is eval overhead and is reported separately from pipeline cost.

A run counts as ERRORED (never as a success; wrong in the primary analysis) if
it raised, if the pipeline set fatal_error, or if any LLM call used a model
other than the pinned generator. Non-fatal entries in the pipeline's `errors`
are kept in `nonfatal_errors`; such runs are scored normally, and the strict
sensitivity analysis (`strict_error`) counts them as wrong as well.

Usage:
  python -m eval_sop.run_conditions --backend anthropic --env-file PATH \
      --conditions a,b,d --n-frames 30 --usd-cap 8 --dry-run
  python -m eval_sop.run_conditions ... --canary          # one known-answer question first
  python -m eval_sop.run_conditions ... --admission-control

Without --admission-control a design whose worst case exceeds the USD cap (or
the Tavily cap) is refused. With it, runs go question by question and a run
starts only if the remaining budget covers that run's own worst case, so the
cap can never be crossed and a partially completed design stays paired.
Exit codes: 0 done, 1 canary failed, 2 refused by the dry-run check,
3 a hard cap or billing stop fired, 4 admission control stopped the design early.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import sys
import traceback
from pathlib import Path

from eval_sop import budget, common, plan

RUNS = common.OUT_DIR / "runs"
# Canary output lives apart from the main runs so it can never enter the
# analysis; `python -m eval_sop.score judge --canary` judges it on its own.
CANARY_RUNS = common.OUT_DIR / "canary_runs"
TOKEN_BUDGET = 80_000  # ReportRequest default — what the deployed API uses
CANARY = {
    "qid": "canary_paris",
    "question": "What is the capital city of France?",
    "reference": "Paris",
    "source": "built-in canary (not part of any benchmark)",
}
COND_NAMES = {
    "a": "closed_book",
    "b": "search1",
    "c": "pipeline_r1",
    "d": "pipeline_r2",
}


def load_questions(n_frames: int, include_open: bool) -> list[dict]:
    qs = [json.loads(line) for line in open(common.HERE / "data" / "questions.jsonl", encoding="utf-8")]
    frames = sorted((q for q in qs if q["qid"].startswith("frames_")), key=lambda q: q["rank"])
    out = frames[:n_frames]
    if include_open:
        out += [q for q in qs if q["qid"].startswith("open_")]
    return out


FINAL_RE = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)


def parse_final(text: str) -> str:
    m = None
    for m in FINAL_RE.finditer(text or ""):
        pass
    if m:
        return m.group(1).strip().strip("*").strip()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


async def run_closed_book(q: dict) -> dict:
    messages = [
        {"role": "system", "content": "You are a careful research assistant."},
        {
            "role": "user",
            "content": (
                f"Question: {q['question']}\n\n"
                "Answer the question. Reason briefly, then give your answer on a "
                "final line that starts with 'Final answer:'."
            ),
        },
    ]
    text, *_ = await common.chat(common.GEN_MODEL, messages, max_tokens=1024, agent="closed_book")
    return {"output_text": text, "final_answer": parse_final(text), "claims": []}


BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


async def run_search1(q: dict, patched) -> dict:
    results = await patched.search(q["question"], max_results=8)
    src = "\n\n".join(
        f"[{i}] {r.title} ({r.url})\n{r.snippet}" for i, r in enumerate(results, 1)
    )
    messages = [
        {"role": "system", "content": "You are a careful research assistant. Use only the provided sources."},
        {
            "role": "user",
            "content": (
                f"Question: {q['question']}\n\nSources:\n{src}\n\n"
                "Write 3-6 bullet points with the key facts from the sources that "
                "bear on the question. End each bullet with the number(s) of the "
                "source(s) it relies on in square brackets, e.g. [2] or [1, 3]. "
                "Then give your answer on a final line that starts with 'Final answer:'."
            ),
        },
    ]
    text, *_ = await common.chat(common.GEN_MODEL, messages, max_tokens=1024, agent="search1")
    claims = []
    for line in text.splitlines():
        m = BULLET_RE.match(line)
        if not m:
            continue
        body = m.group(1)
        nums = [int(n) for g in CITE_RE.findall(body) for n in g.split(",")]
        urls = [results[n - 1].url for n in nums if 1 <= n <= len(results)]
        claim_text = CITE_RE.sub("", body).strip()
        if claim_text:
            claims.append({"text": claim_text, "source_urls": urls, "confidence": None,
                           "supporting_sources": None, "contradicting_sources": None,
                           "context": q["question"], "stage": "final"})
    return {
        "output_text": text,
        "final_answer": parse_final(text),
        "claims": claims,
        "sources": [r.model_dump() for r in results],
        "provided_urls": [r.url for r in results],
    }


def _claim_rows(claims, sub_q: str, round_idx: int) -> list[dict]:
    return [
        {
            "text": c.text,
            "source_urls": c.source_urls,
            "confidence": c.confidence.value,
            "supporting_sources": c.supporting_sources,
            "contradicting_sources": c.contradicting_sources,
            "contradiction_detail": c.contradiction_detail,
            "context": sub_q,
            "round": round_idx,
        }
        for c in claims
    ]


async def run_pipeline_condition(q: dict, max_rounds: int) -> dict:
    from agents.graph import run_pipeline
    from agents.report_format import render_report_markdown
    from agents.schemas import ResearchState

    # model= is recorded on the state for logging; the call itself uses the
    # pinned generator from the per-job creds that common.install() sets.
    state = ResearchState(
        query=q["question"], max_rounds=max_rounds, token_budget=TOKEN_BUDGET,
        model=common.GEN_MODEL,
    )
    rounds: dict[int, dict] = {}

    async def on_progress(s: ResearchState):
        r = rounds.setdefault(s.current_round, {})
        r["sub_questions"] = list(s.sub_questions)
        if s.scraper_outputs:
            r["scraped"] = [
                {"sub_question": o.sub_question,
                 "pages": [{"url": p.url, "ok": bool(p.content and not p.scrape_error),
                            "chars": len(p.content)} for p in o.pages]}
                for o in s.scraper_outputs
            ]
        if s.search_outputs:
            r["search_urls"] = [[x.url for x in o.results] for o in s.search_outputs]
        if s.analyst_outputs:
            r["analyst"] = [
                {"sub_question": a.sub_question, "error": a.error,
                 "contradictions_found": a.contradictions_found,
                 "claims": _claim_rows(a.key_claims, a.sub_question, s.current_round)}
                for a in s.analyst_outputs
            ]
        if s.critic_output and s.activity_log and s.activity_log[-1].agent_name == "Critic Agent":
            r["critic"] = s.critic_output.model_dump(exclude={"approved_claims"})
            r["critic"]["approved_texts"] = [c.text for c in s.critic_output.approved_claims]
            r["critic_round_converged"] = s.converged

    final = await run_pipeline(state, on_progress=on_progress)

    # Claims that reached the Writer = Critic-approved claims (plus FC-verified).
    approved = final.critic_output.approved_claims if final.critic_output else final.collected_claims()
    sub_of = {}
    for r_idx, r in rounds.items():
        for a in r.get("analyst", []):
            for c in a["claims"]:
                sub_of.setdefault(c["text"], (a["sub_question"], r_idx))
    final_claims = []
    for c in approved:
        sq, r_idx = sub_of.get(c.text, (q["question"], None))
        row = _claim_rows([c], sq, r_idx)[0]
        row["stage"] = "final"
        final_claims.append(row)

    md = ""
    if final.final_report:
        try:
            md = render_report_markdown(final.final_report, final.report_mode.value)
        except Exception:  # noqa: BLE001
            md = json.dumps(final.final_report)[:20000]

    # Eval-side answer extraction (not part of the pipeline).
    tok = common.CURRENT_AGENT.set("eval_answer_extraction")
    try:
        messages = [
            {"role": "system", "content": "You answer questions using only the report you are given."},
            {"role": "user", "content": (
                f"Report:\n{md[:24000]}\n\nQuestion: {q['question']}\n\n"
                "Using only the report, answer the question. If the report does not "
                "contain the answer, give your best answer from the report anyway. "
                "Reason briefly, then give your answer on a final line that starts with 'Final answer:'.")},
        ]
        ext_text, *_ = await common.chat(common.GEN_MODEL, messages, max_tokens=512)
    finally:
        common.CURRENT_AGENT.reset(tok)

    provided = sorted({p["url"] for r in rounds.values() for s in r.get("scraped", []) for p in s["pages"] if p["ok"]})
    return {
        "output_text": md,
        "extraction_text": ext_text,
        "final_answer": parse_final(ext_text),
        "claims": final_claims,
        "rounds": rounds,
        "provided_urls": provided,
        "pipeline": {
            "rounds_run": final.current_round + 1,
            "converged": final.converged,
            "fatal_error": final.fatal_error,
            "errors": final.errors,
            "tokens_used_total_pipeline_accounting": final.tokens_used_total,
            "tokens_by_agent": final.tokens_by_agent,
            "fact_check_results": [f.model_dump() for f in final.fact_check_results],
            "quality": (final.final_report or {}).get("quality"),
            "n_citations": len((final.final_report or {}).get("citations", [])),
            "activity": [f"{e.agent_name}: {e.message}" for e in final.activity_log],
        },
    }


async def run_one(cond: str, q: dict, seed: int, patched) -> dict:
    budget.check_stop()
    tel = common.Telemetry()
    t_tok = common.TELEMETRY.set(tel)
    s_tok = common.SEED.set(seed)
    r_tok = common.RUN_KEY.set(f"{COND_NAMES[cond]}|{q['qid']}|{seed}")
    t0 = time.perf_counter()
    try:
        if cond == "a":
            out = await run_closed_book(q)
        elif cond == "b":
            out = await run_search1(q, patched)
        elif cond == "c":
            out = await run_pipeline_condition(q, 1)
        else:
            out = await run_pipeline_condition(q, 2)
        err, err_kind = None, None
    except budget.BudgetStop:
        raise
    except Exception:  # noqa: BLE001
        out, err, err_kind = {}, traceback.format_exc()[-2000:], "exception"
    finally:
        common.TELEMETRY.reset(t_tok)
        common.SEED.reset(s_tok)
        common.RUN_KEY.reset(r_tok)
    # A cap tripped inside the pipeline may have been turned into a value by
    # asyncio.gather(return_exceptions=True); the flag is authoritative. The
    # run is not returned (so not saved): it was cut short, not completed.
    budget.check_stop()
    pipe = out.get("pipeline") if isinstance(out, dict) else None
    if err is None and pipe and pipe.get("fatal_error"):
        err, err_kind = f"pipeline fatal_error: {pipe['fatal_error'][:500]}", "fatal_error"
    # Non-fatal pipeline errors (the Writer's "malformed JSON; raw output
    # preserved", a schema mismatch) leave a usable report: the run is scored
    # normally and flagged, and the strict sensitivity analysis counts it wrong.
    nonfatal = [] if err else list((pipe or {}).get("errors") or [])
    used = {c["model"] for c in tel.llm_calls}
    if err is None and used - {common.GEN_MODEL}:
        err, err_kind = f"unexpected model(s) used: {sorted(used - {common.GEN_MODEL})}", "model_mismatch"
    wall = time.perf_counter() - t0
    tsum = tel.summary()
    pipe_calls = [c for c in tel.llm_calls if c["agent"] != "eval_answer_extraction"]
    tsum["pipeline_only"] = {
        "input_tokens": sum(c["input_tokens"] for c in pipe_calls),
        "output_tokens": sum(c["output_tokens"] for c in pipe_calls),
        "llm_seconds": round(sum(c["seconds"] for c in pipe_calls), 2),
        "llm_calls": len(pipe_calls),
        "truncated_calls": sum(1 for c in pipe_calls if c.get("finish_reason") == "length"),
    }
    return {
        "condition": COND_NAMES[cond],
        "qid": q["qid"],
        "question": q["question"],
        "reference": q.get("reference"),
        "seed": seed,
        "generator": common.GEN_MODEL,
        "wall_seconds_this_invocation": round(wall, 2),
        "telemetry": tsum,
        "error": err,
        "error_kind": err_kind,
        "nonfatal_errors": nonfatal,
        "strict_error": bool(err) or bool(nonfatal),
        **out,
    }


def _ordered(conds: list[str]) -> list[str]:
    """(d) runs before (c) so (c) can replay d's identical round-1 calls from the cache."""
    order = {"a": 0, "b": 1, "d": 2, "c": 3}
    return sorted(conds, key=order.__getitem__)


def _print_plan(table: dict, usd_cap: float, tavily_cap: int, n_q: int, n_seeds: int) -> None:
    print(f"Design: {n_q} questions x {n_seeds} replicate(s), generator {table['model']}")
    for r in table["rows"]:
        print(f"  ({r['condition']}) runs={r['runs']:3d}  worst ${r['worst_usd_per_run']:.4f}/run"
              f"  expected ${r['expected_usd_per_run']:.4f}/run  tavily worst {r['worst_tavily_per_run']}"
              f" expected {r['expected_tavily_per_run']}"
              + ("  (round 1 replayed from d)" if r["shares_round1_with_d"] else ""))
    print(f"  WORST-CASE total ${table['worst_usd_total']:.2f}  (cap ${usd_cap:.2f})")
    print(f"  expected total   ${table['expected_usd_total']:.2f}  (estimate, see eval_sop/plan.py)")
    print(f"  Tavily worst {table['worst_tavily_total']}  expected {table['expected_tavily_total']}"
          f"  (cap {tavily_cap})")


async def _run_canary(args, patched, conds) -> int:
    ok = True
    spent0 = common.LEDGER.spent() if common.LEDGER else 0.0
    for cond in conds:
        rec = await run_one(cond, CANARY, 0, patched)
        path = CANARY_RUNS / COND_NAMES[cond] / f"{CANARY['qid']}__s0.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        good = (not rec["error"]) and "paris" in (rec.get("final_answer") or "").lower()
        ok &= good
        print(f"canary ({cond}) {'OK' if good else 'FAIL'} answer={rec.get('final_answer', '')[:60]!r} "
              f"error={rec['error_kind']} models={rec['telemetry']['models']} "
              f"usd={rec['telemetry']['usd_spent_now']}")
    if common.LEDGER:
        delta = common.LEDGER.spent() - spent0
        print(f"canary ledger delta ${delta:.4f} (0 is expected only if every call replayed from cache)")
    return 0 if ok else 1


async def main_async(args) -> int:
    try:
        patched = common.install(args.backend, args.env_file, args.usd_cap)
    except (ValueError, budget.EvalLocked) as e:  # cap above the hard max; another run active
        print(f"REFUSED: {e}")
        return 2
    common.TAVILY_CREDIT_CAP = args.tavily_cap
    conds = _ordered(args.conditions.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]
    qs = load_questions(args.n_frames, args.open)
    if args.only:
        keep = set(args.only.split(","))
        qs = [q for q in qs if q["qid"] in keep]

    model = common.GEN_MODEL
    paid = args.backend == "anthropic"
    table = plan.design_table(conds, len(qs), len(seeds), model) if paid else None
    if paid:
        _print_plan(table, args.usd_cap, args.tavily_cap, len(qs), len(seeds))
        print(f"  ledger already holds ${common.LEDGER.spent():.4f}; "
              f"Tavily credits already spent by this eval: {common.credits_spent()}")
    if args.dry_run:
        if not paid:
            print("dry run: local backend, no USD cost")
            return 0
        over = (table["worst_usd_total"] > common.LEDGER.remaining()
                or table["worst_tavily_total"] > args.tavily_cap - common.credits_spent())
        if over and not args.admission_control:
            print("REFUSED: worst case exceeds the remaining cap. Shrink the design, or pass "
                  "--admission-control to run question by question within the cap.")
            return 2
        print("dry run OK" + (" (with admission control)" if over else ""))
        return 0
    if paid and not args.admission_control:
        if (table["worst_usd_total"] > common.LEDGER.remaining()
                or table["worst_tavily_total"] > args.tavily_cap - common.credits_spent()):
            print("REFUSED: worst case exceeds the remaining cap (see --dry-run).")
            return 2

    try:
        if args.canary:
            return await _run_canary(args, patched, conds)
        for seed in seeds:
            for q in qs:
                for cond in conds:
                    path = RUNS / COND_NAMES[cond] / f"{q['qid']}__s{seed}.json"
                    if path.exists() and not args.force:
                        prev = json.loads(path.read_text(encoding="utf-8"))
                        if not prev.get("error"):
                            continue  # done; errored runs are retried (cached calls replay free)
                    if paid:
                        shared = cond == "c" and "d" in conds
                        need = plan.worst_usd(cond, model, shared)
                        need_t = plan.worst_tavily(cond, shared)
                        if (common.LEDGER.remaining() < need
                                or args.tavily_cap - common.credits_spent() < need_t):
                            print(f"STOP (admission): next run ({cond}) {q['qid']} worst case "
                                  f"${need:.3f} / {need_t} credits does not fit the remaining budget")
                            return 4  # non-zero: the planned design did not complete
                    path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[{time.strftime('%H:%M:%S')}] {COND_NAMES[cond]} {q['qid']} seed={seed} "
                          f"usd_spent={common.LEDGER.spent() if paid else 0:.4f} "
                          f"credits_spent={common.credits_spent()}", flush=True)
                    rec = await run_one(cond, q, seed, patched)
                    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str),
                                    encoding="utf-8")
                    t = rec["telemetry"]
                    print(f"    -> answer={rec.get('final_answer', '')[:80]!r} err={rec['error_kind']} "
                          f"usd={t['usd_spent_now']} credits_now={t['tavily_credits_spent_now']}",
                          flush=True)
    except budget.BudgetStop as e:
        print(f"STOP: {e}", flush=True)
        return 3
    finally:
        if common.LEDGER:
            common.LEDGER.export_jsonl(common.OUT_DIR / "results" / "usd_ledger.jsonl")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(common.GEN_MODELS), default="ollama")
    ap.add_argument("--env-file", type=Path, default=None,
                    help="file holding TAVILY_API_KEY / ANTHROPIC_API_KEY (read in-process, never printed)")
    ap.add_argument("--conditions", default="a,b,d")
    ap.add_argument("--seeds", default="1", help="replicate indices; not sampling seeds on Anthropic")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--usd-cap", type=float, default=8.0)
    ap.add_argument("--tavily-cap", type=int, default=common.TAVILY_CREDIT_CAP)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--admission-control", action="store_true")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
