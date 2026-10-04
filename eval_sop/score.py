"""
Score saved runs.

  python -m eval_sop.score judge     # repo venv: LLM correctness judges + fetch cited pages
  python -m eval_sop.score judge --canary   # same, on the canary output only
  python -m eval_sop.score support   # torch env: NLI citation-support judges
  python -m eval_sop.score analyze   # pure computation -> results/*.json, CSV

Correctness labels are LLM-judge labels (llama3.1:8b and gemma2:9b, local),
plus `strict_match`, a deterministic string check. Citation-support labels come
from two NLI models (eval_sop/nli.py) whose agreement with human labels is
measured on RAGTruth by nli_validate.py.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import random
import re
import string
import sys
from collections import defaultdict

from eval_sop import common, stats
from eval_sop.judges import judge_correct
from eval_sop.quality_metrics import (
    contradiction_rate,
    domain_diversity,
    structural_coverage,
)

RUNS = common.OUT_DIR / "runs"
RES = common.OUT_DIR / "results"
JUDGED = RES / "judgments.jsonl"
CLAIMS_PER_RUN = 6
URLS_PER_CLAIM = 3
CONF_ORD = {"high": 3, "medium": 2, "low": 1, "contested": 0, "inconclusive": 0}
SUPPORT_JUDGES = ("deberta-v3-large", "deberta-xlarge-mnli")  # eval_sop/nli.py MODELS
SONNET_PRICE = {"input": 3.00, "output": 15.00}  # claude-sonnet-4-5, $/1M tokens


def load_runs() -> list[dict]:
    out = []
    for p in sorted(RUNS.glob("*/*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def run_key(r) -> str:
    return f"{r['condition']}|{r['qid']}|{r['seed']}"


def sample_claims(r) -> list[dict]:
    claims = [c for c in r.get("claims", []) if c.get("text")]
    rng = random.Random(int(hashlib.sha256(run_key(r).encode()).hexdigest()[:8], 16))
    if len(claims) > CLAIMS_PER_RUN:
        claims = rng.sample(claims, CLAIMS_PER_RUN)
    return claims


# ---------------------------------------------------------------------------
# Deterministic answer check
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def strict_match(reference: str, prediction: str) -> bool:
    ref, pred = _norm(reference), _norm(prediction)
    return bool(ref) and (ref in pred)


# ---------------------------------------------------------------------------
# Judge phase
# ---------------------------------------------------------------------------
async def judge_all():
    patched = common.install()
    done = set()
    if JUDGED.exists():
        for line in open(JUDGED, encoding="utf-8"):
            done.add(json.loads(line)["key"])
    RES.mkdir(exist_ok=True)
    runs = load_runs()
    out = open(JUDGED, "a", encoding="utf-8")
    # Order by judge model so Ollama swaps models as rarely as possible.
    for model in common.JUDGE_MODELS:
        for r in runs:
            if r.get("error"):
                continue
            if r.get("reference"):
                key = f"correct|{model}|{run_key(r)}"
                if key not in done:
                    j = await judge_correct(model, r["question"], r["reference"], r.get("final_answer", ""))
                    out.write(json.dumps({"key": key, "kind": "correct", "judge": model,
                                          "run": run_key(r), **j}) + "\n")
                    out.flush()
                    done.add(key)
            for c in sample_claims(r):
                for u in (c.get("source_urls") or [])[:URLS_PER_CLAIM]:
                    if patched.pages.get(u) is None:
                        try:
                            await patched.scrape(u, "")
                        except Exception:  # noqa: BLE001 - recorded as missing evidence
                            pass
            print(f"judged {model} {run_key(r)}", flush=True)
    out.close()


def support_nli():
    """NLI support judgments for every sampled (claim, cited URL) pair."""
    from eval_sop.nli import MODELS, NLIJudge

    pages = common.KV("pages")
    done = set()
    if JUDGED.exists():
        done = {json.loads(line)["key"] for line in open(JUDGED, encoding="utf-8")}
    runs = [r for r in load_runs() if not r.get("error")]
    with open(JUDGED, "a", encoding="utf-8") as out:
        for name in MODELS:
            judge = None
            for r in runs:
                for c in sample_claims(r):
                    for u in (c.get("source_urls") or [])[:URLS_PER_CLAIM]:
                        key = f"support|{name}|{run_key(r)}|{hashlib.sha1((c['text'] + u).encode()).hexdigest()}"
                        if key in done:
                            continue
                        content = ((pages.get(u) or {}).get("content") or "").strip()
                        if content:
                            judge = judge or NLIJudge(name)
                            s = judge.score(content, c["text"])
                        else:
                            s = {"p_entail": None, "supported": None, "n_windows": 0}
                        out.write(json.dumps({"key": key, "kind": "support", "judge": name,
                                              "run": run_key(r), "claim": c["text"], "url": u,
                                              "evidence_missing": not content, **s}) + chr(10))
                        out.flush()
                        done.add(key)


# ---------------------------------------------------------------------------
# Analyze phase
# ---------------------------------------------------------------------------
def _reviewed_claims_by_round(r) -> dict[int, dict]:
    """Reconstruct what each Critic call reviewed: carried approved + this round's claims."""
    rounds = {int(k): v for k, v in (r.get("rounds") or {}).items()}
    by_text = {}
    out = {}
    carried: list[dict] = []
    for idx in sorted(rounds):
        rd = rounds[idx]
        current = [c for a in rd.get("analyst", []) for c in a["claims"]]
        for c in current:
            by_text.setdefault(c["text"], c)
        seen = {c["text"] for c in carried}
        reviewed = list(carried) + [c for c in current if c["text"] not in seen]
        out[idx] = {"reviewed": reviewed, "sub_questions": rd.get("sub_questions", []),
                    "critic": rd.get("critic")}
        if rd.get("critic"):
            for t in rd["critic"].get("approved_texts", []):
                if t in by_text and t not in {c["text"] for c in carried}:
                    carried.append(by_text[t])
    return out


def analyze():
    runs = [r for r in load_runs()]
    judg = [json.loads(line) for line in open(JUDGED, encoding="utf-8")] if JUDGED.exists() else []
    correct = defaultdict(dict)  # run -> judge -> bool
    support = defaultdict(lambda: defaultdict(dict))  # run -> claim -> judge -> list
    missing = defaultdict(dict)
    for j in judg:
        if j["kind"] == "correct":
            correct[j["run"]][j["judge"]] = bool(j["correct"])
        else:
            support[j["run"]][j["claim"]].setdefault(j["judge"], []).append(j["supported"])
            missing[j["run"]].setdefault(j["claim"], []).append(j["evidence_missing"])

    J1, J2 = common.JUDGE_MODELS  # correctness (LLM judges)
    S1, S2 = SUPPORT_JUDGES  # citation support (NLI)
    conds = ["closed_book", "search1", "pipeline_r1", "pipeline_r2"]
    per_run = []
    claim_rows = []
    for r in runs:
        k = run_key(r)
        row = {"run": k, "condition": r["condition"], "qid": r["qid"], "seed": r["seed"],
               "error": bool(r.get("error")), "has_ref": bool(r.get("reference")),
               "strict_error": bool(r.get("strict_error", r.get("error")))}
        if r.get("reference") and not r.get("error"):
            row["strict_match"] = strict_match(r["reference"], r.get("final_answer", ""))
            for jm in (J1, J2):
                if jm in correct[k]:
                    row[f"correct_{jm}"] = correct[k][jm]
            if J1 in correct[k] and J2 in correct[k]:
                row["correct_both"] = correct[k][J1] and correct[k][J2]
        # claim support for this run
        provided = set(r.get("provided_urls") or [])
        sup_by_judge = {S1: [], S2: []}
        for c in sample_claims(r) if not r.get("error") else []:
            labels = support[k].get(c["text"], {})
            urls = (c.get("source_urls") or [])[:URLS_PER_CLAIM]
            cr = {"run": k, "condition": r["condition"], "qid": r["qid"], "seed": r["seed"],
                  "claim": c["text"], "context": c.get("context"), "urls": urls,
                  "confidence": c.get("confidence"), "supporting_sources": c.get("supporting_sources"),
                  "contradicting_sources": c.get("contradicting_sources"),
                  "n_urls": len(c.get("source_urls") or []),
                  "all_urls_were_provided": bool(urls) and all(u in provided for u in urls),
                  "any_evidence_missing": any(missing[k].get(c["text"], [])),
                  "all_evidence_missing": bool(urls) and all(missing[k].get(c["text"], [True]))}
            for jm in (S1, S2):
                vals = labels.get(jm, [])
                # Supported if ANY cited page is judged to support it; a claim
                # with no citation, or no fetchable cited page, is unsupported.
                cr[f"supported_{jm}"] = any(v is True for v in vals) if urls else False
                if urls and not vals:
                    cr[f"supported_{jm}"] = None  # not judged yet
                if cr[f"supported_{jm}"] is not None:
                    sup_by_judge[jm].append(cr[f"supported_{jm}"])
            if cr.get(f"supported_{S1}") is not None and cr.get(f"supported_{S2}") is not None:
                cr["supported_both"] = cr[f"supported_{S1}"] and cr[f"supported_{S2}"]
            claim_rows.append(cr)
        for jm in (S1, S2):
            v = sup_by_judge[jm]
            row[f"support_rate_{jm}"] = sum(v) / len(v) if v else None
        row["n_claims_total"] = len(r.get("claims", []))
        t = r.get("telemetry", {})
        po = t.get("pipeline_only", {})
        row.update({
            "in_tok": po.get("input_tokens"), "out_tok": po.get("output_tokens"),
            "llm_seconds": po.get("llm_seconds"), "llm_calls": po.get("llm_calls"),
            "truncated_calls": po.get("truncated_calls"),
            "tavily_credits": t.get("tavily_credits_if_uncached"),
        })
        row["est_cost_sonnet45_usd"] = (
            (po.get("input_tokens") or 0) / 1e6 * SONNET_PRICE["input"]
            + (po.get("output_tokens") or 0) / 1e6 * SONNET_PRICE["output"]
        )
        pipe = r.get("pipeline")
        if pipe:
            row.update({
                "rounds_run": pipe["rounds_run"], "converged": pipe["converged"],
                "fatal_error": pipe["fatal_error"], "n_errors": len(pipe["errors"]),
                "writer_failed_parse": any("malformed JSON" in e for e in pipe["errors"]),
                "schema_mismatch": any("did not match" in e for e in pipe["errors"]),
                "n_fact_checks": len(pipe["fact_check_results"]),
                "fc_verdicts": [f["verdict"] for f in pipe["fact_check_results"]],
                "tokens_pipeline_accounting": pipe["tokens_used_total_pipeline_accounting"],
                "over_budget": pipe["tokens_used_total_pipeline_accounting"] > 80_000,
            })
            # Critic self-report vs computed, per Critic call
            crit_rows = []
            for idx, rv in _reviewed_claims_by_round(r).items():
                cr = rv["critic"]
                if not cr:
                    continue
                dd = domain_diversity(rv["reviewed"])
                crit_rows.append({
                    "round": idx,
                    "n_reviewed": len(rv["reviewed"]),
                    "n_flagged": len(cr.get("flagged_claims", [])),
                    "critic_diversity": cr["source_diversity_score"],
                    "computed_diversity_unique_ratio": dd["unique_ratio"],
                    "computed_diversity_entropy": dd["entropy_norm"],
                    "n_domains": dd["n_domains"],
                    "critic_contradiction_rate": cr["contradiction_rate"],
                    "computed_contradiction_rate": contradiction_rate(rv["reviewed"]),
                    "critic_coverage": cr["coverage_score"],
                    "computed_structural_coverage": structural_coverage(
                        rv["sub_questions"], [c for c in rv["reviewed"]]),
                    "critic_quality": cr["overall_quality_score"],
                    "critic_needs_more": cr["needs_more_research"],
                    "critic_notes_failed": str(cr.get("critic_notes", "")).startswith("Critic failed"),
                })
            row["critic_calls"] = crit_rows
            if crit_rows:
                last = crit_rows[-1]
                row["final_critic_quality"] = last["critic_quality"]
                row["final_critic_coverage"] = last["critic_coverage"]
        per_run.append(row)

    summary = {"n_runs": len(per_run), "conditions": {}}
    # --- Per-condition aggregates (unit = question; seeds averaged per question)
    def by_question(cond, field, only_ref=False):
        acc = defaultdict(list)
        for row in per_run:
            if row["condition"] != cond or row["error"]:
                continue
            if only_ref and not row["has_ref"]:
                continue
            v = row.get(field)
            if v is None:
                continue
            acc[row["qid"]].append(float(v))
        return {q: sum(v) / len(v) for q, v in acc.items()}

    def per_seed(cond, field, only_ref=False):
        acc = defaultdict(list)
        for row in per_run:
            if row["condition"] == cond and not row["error"] and row.get(field) is not None:
                if only_ref and not row["has_ref"]:
                    continue
                acc[row["seed"]].append(float(row[field]))
        return {s: sum(v) / len(v) for s, v in acc.items()}

    metrics = {
        "strict_match": True, f"correct_{J1}": True, f"correct_{J2}": True, "correct_both": True,
        f"support_rate_{S1}": False, f"support_rate_{S2}": False,
        "in_tok": False, "out_tok": False, "llm_seconds": False, "llm_calls": False,
        "tavily_credits": False, "est_cost_sonnet45_usd": False,
    }
    for cond in conds:
        cs = {"n_runs": sum(1 for r in per_run if r["condition"] == cond),
              "n_errors": sum(1 for r in per_run if r["condition"] == cond and r["error"]),
              "seeds": sorted({r["seed"] for r in per_run if r["condition"] == cond})}
        for m, only_ref in metrics.items():
            bq = by_question(cond, m, only_ref)
            ps = per_seed(cond, m, only_ref)
            agg = stats.mean_ci(list(bq.values()))
            agg["per_seed_means"] = ps
            agg["std_across_seeds"] = (
                float(stats.np.std(list(ps.values()), ddof=1)) if len(ps) > 1 else None)
            cs[m] = agg
        rows = [r for r in per_run if r["condition"] == cond and not r["error"]]
        if cond.startswith("pipeline"):
            cs["pipeline_health"] = {
                "rounds_run_mean": _mean([r.get("rounds_run") for r in rows]),
                "converged_rate": _mean([r.get("converged") for r in rows]),
                "fatal_error_rate": _mean([bool(r.get("fatal_error")) for r in rows]),
                "writer_unparseable_rate": _mean([r.get("writer_failed_parse") for r in rows]),
                "schema_mismatch_rate": _mean([r.get("schema_mismatch") for r in rows]),
                "runs_with_any_fact_check": _mean([r.get("n_fact_checks", 0) > 0 for r in rows]),
                "fact_checks_total": sum(r.get("n_fact_checks", 0) for r in rows),
                "fc_verdict_counts": _count([v for r in rows for v in r.get("fc_verdicts", [])]),
                "over_80k_token_budget_rate": _mean([r.get("over_budget") for r in rows]),
                "runs_with_truncated_llm_call": _mean([(r.get("truncated_calls") or 0) > 0 for r in rows]),
                "final_claims_mean": _mean([r.get("n_claims_total") for r in rows]),
            }
        summary["conditions"][cond] = cs

    # --- Paired differences (unit = question)
    pairs = [("closed_book", "search1"), ("search1", "pipeline_r1"),
             ("pipeline_r1", "pipeline_r2"), ("search1", "pipeline_r2"),
             ("closed_book", "pipeline_r2")]
    summary["paired_differences"] = {}
    for a, b in pairs:
        d = {}
        for m in ["strict_match", f"correct_{J1}", f"correct_{J2}", f"support_rate_{S1}", f"support_rate_{S2}"]:
            only_ref = m.startswith("correct") or m == "strict_match"
            d[m] = stats.paired_diff_ci(by_question(a, m, only_ref), by_question(b, m, only_ref))
        summary["paired_differences"][f"{b} minus {a}"] = d

    # --- Primary test: exact McNemar on per-question correctness, single
    # replicate (seed 1), intention-to-treat: an errored run counts as wrong.
    def itt(cond, field, seed=1, err_key="error"):
        out = {}
        for row in per_run:
            if row["condition"] != cond or not row["has_ref"] or row["seed"] != seed:
                continue
            out[row["qid"]] = 0 if row[err_key] else int(bool(row.get(field)))
        return out

    ms = [f"correct_{J1}", f"correct_{J2}", "strict_match"]
    summary["mcnemar_itt"] = {
        f"{b} vs {a}": {m: stats.mcnemar_exact(itt(a, m), itt(b, m)) for m in ms}
        for a, b in pairs
    }
    summary["mcnemar_itt_strict"] = {
        f"{b} vs {a}": {m: stats.mcnemar_exact(itt(a, m, err_key="strict_error"),
                                               itt(b, m, err_key="strict_error")) for m in ms}
        for a, b in pairs
    }
    summary["mcnemar_itt_note"] = (
        "PRIMARY: a run counts as wrong only if it raised, set fatal_error, or used a "
        "non-pinned model; runs with non-fatal pipeline errors are scored normally. "
        "STRICT (sensitivity): those runs also count as wrong. Correctness labels are "
        "local LLM-judge labels except strict_match (deterministic).")

    # --- Critic self-report vs computed
    crit = [dict(c, condition=r["condition"], run=r["run"]) for r in per_run for c in r.get("critic_calls", [])]
    def _pair(xk, yk):
        xs = [(c[xk], c[yk]) for c in crit if c.get(xk) is not None and c.get(yk) is not None]
        if not xs:
            return {"n": 0}
        x, y = zip(*xs)
        return {"n": len(xs), "mean_self_reported": _mean(x), "mean_computed": _mean(y),
                "mae": _mean([abs(a - b) for a, b in xs]), "spearman": stats.spearman(x, y)}
    summary["critic_vs_computed"] = {
        "n_critic_calls": len(crit),
        "diversity_vs_unique_ratio": _pair("critic_diversity", "computed_diversity_unique_ratio"),
        "diversity_vs_entropy": _pair("critic_diversity", "computed_diversity_entropy"),
        "contradiction_rate": _pair("critic_contradiction_rate", "computed_contradiction_rate"),
        "coverage_vs_structural": _pair("critic_coverage", "computed_structural_coverage"),
        "calls_reporting_contradictions_when_none_flagged_in_data": sum(
            1 for c in crit if (c["critic_contradiction_rate"] or 0) > 0 and c["computed_contradiction_rate"] == 0),
        "calls_with_zero_computed_contradictions": sum(1 for c in crit if c["computed_contradiction_rate"] == 0),
        "critic_llm_failed_calls": sum(1 for c in crit if c["critic_notes_failed"]),
    }

    # --- Calibration
    cal = {}
    pc = [c for c in claim_rows if c["condition"].startswith("pipeline") and c.get("confidence")]
    for jm in (S1, S2, "both"):
        key = f"supported_{jm}"
        rows = [c for c in pc if c.get(key) is not None]
        y = [int(c[key]) for c in rows]
        cal[f"claim_confidence_vs_{key}"] = {
            "n": len(rows),
            "auroc_confidence_ordinal": stats.auroc(y, [CONF_ORD[c["confidence"]] for c in rows]),
            "auroc_ci95": stats.auroc_ci(y, [CONF_ORD[c["confidence"]] for c in rows]),
            "auroc_supporting_count": stats.auroc(y, [c["supporting_sources"] or 0 for c in rows]),
            "spearman_confidence": stats.spearman([CONF_ORD[c["confidence"]] for c in rows], y),
            "support_rate_by_label": {
                lab: _mean([c[key] for c in rows if c["confidence"] == lab])
                for lab in ["high", "medium", "low", "contested"]},
            "n_by_label": _count([c["confidence"] for c in rows]),
        }
    pr = [r for r in per_run if r["condition"].startswith("pipeline") and r.get("final_critic_quality") is not None]
    for jm in (J1, J2):
        rows = [r for r in pr if r.get(f"correct_{jm}") is not None]
        y = [int(r[f"correct_{jm}"]) for r in rows]
        cal[f"critic_quality_vs_correct_{jm}"] = {
            "n": len(rows), "n_correct": sum(y),
            "auroc": stats.auroc(y, [r["final_critic_quality"] for r in rows]),
            "auroc_ci95": stats.auroc_ci(y, [r["final_critic_quality"] for r in rows]),
        }
        cal[f"critic_coverage_vs_correct_{jm}"] = {
            "n": len(rows),
            "auroc": stats.auroc(y, [r["final_critic_coverage"] for r in rows]),
            "auroc_ci95": stats.auroc_ci(y, [r["final_critic_coverage"] for r in rows]),
        }
    for sm in (S1, S2):
        rows2 = [r for r in pr if r.get(f"support_rate_{sm}") is not None]
        cal[f"critic_quality_vs_support_rate_{sm}"] = stats.spearman(
            [r["final_critic_quality"] for r in rows2], [r[f"support_rate_{sm}"] for r in rows2])
    summary["calibration"] = cal

    # --- Judge agreement on this eval's own items
    both_c = [r for r in per_run if r.get(f"correct_{J1}") is not None and r.get(f"correct_{J2}") is not None]
    both_s = [c for c in claim_rows if c.get(f"supported_{S1}") is not None and c.get(f"supported_{S2}") is not None]
    summary["judge_agreement_on_eval_items"] = {
        "correctness": {"n": len(both_c),
                        "kappa": stats.cohen_kappa([r[f"correct_{J1}"] for r in both_c], [r[f"correct_{J2}"] for r in both_c]) if both_c else None,
                        "kappa_judge1_vs_strict_match": stats.cohen_kappa([r[f"correct_{J1}"] for r in both_c], [r["strict_match"] for r in both_c]) if both_c else None,
                        "kappa_judge2_vs_strict_match": stats.cohen_kappa([r[f"correct_{J2}"] for r in both_c], [r["strict_match"] for r in both_c]) if both_c else None},
        "support": {"n": len(both_s),
                    "kappa": stats.cohen_kappa([c[f"supported_{S1}"] for c in both_s], [c[f"supported_{S2}"] for c in both_s]) if both_s else None},
    }
    summary["claims"] = {
        cond: {
            "n_judged_claims": sum(1 for c in claim_rows if c["condition"] == cond),
            "claims_without_citation": _mean([c["n_urls"] == 0 for c in claim_rows if c["condition"] == cond]),
            "cited_url_not_among_fetched_sources": _mean([not c["all_urls_were_provided"] for c in claim_rows if c["condition"] == cond and c["n_urls"]]),
            "all_cited_pages_unfetchable": _mean([c["all_evidence_missing"] for c in claim_rows if c["condition"] == cond and c["n_urls"]]),
        } for cond in conds
    }

    RES.mkdir(exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (RES / "per_run.json").write_text(json.dumps(per_run, indent=1, default=str), encoding="utf-8")
    (RES / "claims_judged.json").write_text(json.dumps(claim_rows, indent=1, default=str), encoding="utf-8")
    export_csv(claim_rows)
    print(json.dumps(summary, indent=1, default=str)[:6000])


def export_csv(claim_rows):
    """Optional artifact: ~100 claims for anyone who later wants to hand-check the judges."""
    pages = common.KV("pages")
    rng = random.Random(99)
    pool = [c for c in claim_rows if c["urls"]]
    pick = rng.sample(pool, min(100, len(pool)))
    S1, S2 = SUPPORT_JUDGES
    with open(RES / "claims_for_optional_human_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "qid", "seed", "context", "claim", "cited_url", "page_excerpt_first_600_chars",
                    f"nli_judge_{S1}", f"nli_judge_{S2}", "pipeline_confidence_label", "human_label (optional, blank)"])
        for c in pick:
            u = c["urls"][0]
            page = pages.get(u) or {}
            w.writerow([c["condition"], c["qid"], c["seed"], c["context"], c["claim"], u,
                        (page.get("content") or "")[:600].replace("\n", " "),
                        c.get(f"supported_{S1}"), c.get(f"supported_{S2}"), c.get("confidence"), ""])


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


if __name__ == "__main__":
    if "--canary" in sys.argv:
        # Judge only the canary output (written by run_conditions --canary),
        # into a separate results folder: checks the local judges end to end
        # after the canary and before the main paid run.
        RUNS = common.OUT_DIR / "canary_runs"
        RES = common.OUT_DIR / "results_canary"
        JUDGED = RES / "judgments.jsonl"
    if sys.argv[1] == "judge":
        asyncio.run(judge_all())
    elif sys.argv[1] == "support":
        support_nli()
    else:
        analyze()
