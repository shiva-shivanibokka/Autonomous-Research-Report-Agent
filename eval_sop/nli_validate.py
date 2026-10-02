"""
Validate the NLI support judges against HUMAN labels from RAGTruth.

Source: the RAGTruth test split (test.parquet, 2,700 responses, span-level human
hallucination annotations) copied read-only from another local eval's cache
into eval_sop/cache/ragtruth/ (sha256 recorded in RESULTS.md). Not committed.

Item construction (fixed before running any judge):
* tasks QA and Summary only (Data2txt contexts are JSON tables, unlike web pages);
* responses with quality == "good";
* each response is split into sentences; a sentence is UNSUPPORTED if it
  overlaps any human-annotated hallucination span (any label type), else SUPPORTED;
* sentences shorter than 25 characters are dropped;
* random.Random(2026) draws 75 supported + 75 unsupported sentences per task
  -> 300 balanced items. Premise = the response's source context.
* SECOND_JUDGE_PER_CELL can restrict the slower second judge to a subset;
  in the reported run it is 75, i.e. both judges see all 300 items.

  python -m eval_sop.nli_validate            (uses OMP_NUM_THREADS=2)
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

from eval_sop import stats
from eval_sop.nli import MODELS, NLIJudge

HERE = Path(__file__).resolve().parent
SRC = HERE / "cache" / "ragtruth" / "test.parquet"
OUT_RAW = HERE / "results" / "nli_ragtruth_raw.jsonl"
OUT = HERE / "results" / "nli_ragtruth_validation.json"
SENT = re.compile(r"[^.!?]+(?:[.!?]+|$)")


def build_items() -> list[dict]:
    df = pd.read_parquet(SRC)
    df = df[df.task_type.isin(["QA", "Summary"]) & (df.quality == "good")]
    pools: dict[tuple[str, bool], list[dict]] = {}
    for _, r in df.iterrows():
        spans = [(s["start"], s["end"]) for s in json.loads(r.hallucination_labels)]
        for m in SENT.finditer(r.output):
            text = m.group(0).strip()
            if len(text) < 25:
                continue
            a, b = m.start(), m.end()
            bad = any(s < b and e > a for s, e in spans)
            pools.setdefault((r.task_type, not bad), []).append(
                {"id": f"{r.id}:{a}", "task": r.task_type, "claim": text,
                 "evidence": r.context, "human_supported": not bad}
            )
    rng = random.Random(2026)
    items = []
    for key in sorted(pools):
        items += rng.sample(pools[key], 75)
    return items


SECOND_JUDGE_PER_CELL = 75  # = all items; lower it to lighten CPU use


def items_for(name: str, items: list[dict]) -> list[dict]:
    if name == list(MODELS)[0]:
        return items
    out, seen = [], {}
    for it in items:  # items are grouped by cell in build order
        key = (it["task"], it["human_supported"])
        if seen.get(key, 0) < SECOND_JUDGE_PER_CELL:
            out.append(it)
            seen[key] = seen.get(key, 0) + 1
    return out


def main():
    items = build_items()
    OUT_RAW.parent.mkdir(exist_ok=True)
    done = set()
    if OUT_RAW.exists():
        done = {(j["judge"], j["id"]) for j in map(json.loads, open(OUT_RAW, encoding="utf-8"))}
    with open(OUT_RAW, "a", encoding="utf-8") as f:
        for name in MODELS:
            mine = items_for(name, items)
            if all((name, it["id"]) in done for it in mine):
                continue
            judge = NLIJudge(name)
            t0 = time.time()
            for i, it in enumerate(mine):
                if (name, it["id"]) in done:
                    continue
                s = judge.score(it["evidence"], it["claim"])
                f.write(json.dumps({"judge": name, "id": it["id"], "task": it["task"],
                                    "human_supported": it["human_supported"], **s}) + "\n")
                f.flush()
                if i % 25 == 0:
                    print(name, i, round(time.time() - t0), "s", flush=True)
            del judge
    summarize(items)


def summarize(items):
    rows = [json.loads(line) for line in open(OUT_RAW, encoding="utf-8")]
    ids = {it["id"] for it in items}
    rep = {"n_items": len(items), "construction": __doc__.split("Item construction")[1].split("python -m")[0].strip(),
           "judges": {}}
    for name in MODELS:
        mine = {it["id"] for it in items_for(name, items)}
        rs = sorted((r for r in rows if r["judge"] == name and r["id"] in mine), key=lambda r: r["id"])
        y = [r["human_supported"] for r in rs]
        p = [r["supported"] for r in rs]
        rep["judges"][name] = {
            "n": len(rs),
            "balanced_accuracy": stats.balanced_accuracy(y, p),
            "balanced_accuracy_ci95": stats.bootstrap_ci_pairs(y, p, stats.balanced_accuracy, boot=2000),
            "cohen_kappa_vs_human": stats.cohen_kappa(y, p),
            "kappa_ci95": stats.bootstrap_ci_pairs(y, p, stats.cohen_kappa, boot=2000),
            "auroc_p_entail": stats.auroc(y, [r["p_entail"] for r in rs]),
            "auroc_p_entail_ci95": stats.auroc_ci(y, [r["p_entail"] for r in rs]),
            "predicted_supported_rate": sum(p) / len(p) if p else None,
            "recall_supported": sum(a and b for a, b in zip(y, p)) / max(sum(y), 1),
            "recall_unsupported": sum((not a) and (not b) for a, b in zip(y, p)) / max(len(y) - sum(y), 1),
            "per_task_balanced_accuracy": {
                t: stats.balanced_accuracy([r["human_supported"] for r in rs if r["task"] == t],
                                           [r["supported"] for r in rs if r["task"] == t])
                for t in ("QA", "Summary")},
        }
    names = list(MODELS)
    a = {r["id"]: r["supported"] for r in rows if r["judge"] == names[0]}
    b = {r["id"]: r["supported"] for r in rows if r["judge"] == names[1]}
    common_ids = sorted(set(a) & set(b) & ids)
    if common_ids:
        rep["inter_judge"] = {
            "n": len(common_ids),
            "cohen_kappa": stats.cohen_kappa([a[i] for i in common_ids], [b[i] for i in common_ids]),
            "raw_agreement": sum(a[i] == b[i] for i in common_ids) / len(common_ids),
        }
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(build_items())
    else:
        main()
