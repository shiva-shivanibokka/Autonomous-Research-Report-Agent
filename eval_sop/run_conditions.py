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

Usage:
  python -m eval_sop.run_conditions --conditions a,b,c,d --seeds 1 --n-frames 30 --open
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import traceback
from pathlib import Path

from eval_sop import common

RUNS = common.OUT_DIR / "runs"
TOKEN_BUDGET = 80_000  # ReportRequest default — what the deployed API uses
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

    state = ResearchState(query=q["question"], max_rounds=max_rounds, token_budget=TOKEN_BUDGET)
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
    tel = common.Telemetry()
    t_tok = common.TELEMETRY.set(tel)
    s_tok = common.SEED.set(seed)
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
        err = None
    except common.CreditCapReached:
        raise
    except Exception as e:  # noqa: BLE001
        out, err = {}, traceback.format_exc()[-2000:]
    finally:
        common.TELEMETRY.reset(t_tok)
        common.SEED.reset(s_tok)
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
        **out,
    }


async def main_async(args):
    patched = common.install()
    qs = load_questions(args.n_frames, args.open)
    if args.only:
        keep = set(args.only.split(","))
        qs = [q for q in qs if q["qid"] in keep]
    conds = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        for q in qs:
            for cond in conds:
                path = RUNS / COND_NAMES[cond] / f"{q['qid']}__s{seed}.json"
                if path.exists() and not args.force:
                    prev = json.loads(path.read_text(encoding="utf-8"))
                    if not prev.get("error"):
                        continue  # done; errored runs are retried
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[{time.strftime('%H:%M:%S')}] {COND_NAMES[cond]} {q['qid']} seed={seed} "
                      f"credits_spent={common.credits_spent()}", flush=True)
                try:
                    rec = await run_one(cond, q, seed, patched)
                except common.CreditCapReached as e:
                    print(f"STOP: {e}", flush=True)
                    return
                path.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                t = rec["telemetry"]
                print(f"    -> answer={rec.get('final_answer', '')[:80]!r} err={bool(rec['error'])} "
                      f"llm_s={t['llm_seconds']} wall={rec['wall_seconds_this_invocation']} "
                      f"credits_now={t['tavily_credits_spent_now']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="a,b,c,d")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
