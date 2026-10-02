"""
Validate the citation-support judges against HUMAN labels from AttributionBench
(osunlp/AttributionBench, test split, 320-item balanced sample; see
build_datasets.py). Reports balanced accuracy and Cohen's kappa per judge, per
source dataset, and the agreement between the two judges.

  python -m eval_sop.validate_judge
"""

from __future__ import annotations

import asyncio
import json

from eval_sop import common, stats
from eval_sop.judges import judge_support

OUT = common.HERE / "results" / "judge_validation.json"


async def main_async():
    common.install()
    items = [
        json.loads(line)
        for line in open(common.HERE / "data" / "attributionbench_320.jsonl", encoding="utf-8")
    ]
    rows = []
    for model in common.JUDGE_MODELS:
        for i, it in enumerate(items):
            j = await judge_support(model, it["question"], it["claim"], it["evidence"])
            rows.append(
                {
                    "id": it["id"],
                    "src_dataset": it["src_dataset"],
                    "judge": model,
                    "human": it["human_label"] == "attributable",
                    "pred": j["supported"],
                    "raw": j["raw"],
                }
            )
            if i % 40 == 0:
                print(model, i, flush=True)

    report = {"n_items": len(items), "judges": {}}
    for model in common.JUDGE_MODELS:
        rs = [r for r in rows if r["judge"] == model]
        y = [r["human"] for r in rs]
        p = [bool(r["pred"]) for r in rs]  # unparsable -> not supported
        per_src = {}
        for src in sorted({r["src_dataset"] for r in rs}):
            sub = [r for r in rs if r["src_dataset"] == src]
            per_src[src] = {
                "n": len(sub),
                "balanced_accuracy": stats.balanced_accuracy(
                    [r["human"] for r in sub], [bool(r["pred"]) for r in sub]
                ),
            }
        report["judges"][model] = {
            "balanced_accuracy": stats.balanced_accuracy(y, p),
            "balanced_accuracy_ci95": stats.bootstrap_ci_pairs(y, p, stats.balanced_accuracy),
            "cohen_kappa_vs_human": stats.cohen_kappa(y, p),
            "predicted_positive_rate": sum(p) / len(p),
            "unparsable": sum(r["pred"] is None for r in rs),
            "per_source": per_src,
        }
    a = {r["id"]: bool(r["pred"]) for r in rows if r["judge"] == common.JUDGE_MODELS[0]}
    b = {r["id"]: bool(r["pred"]) for r in rows if r["judge"] == common.JUDGE_MODELS[1]}
    ids = sorted(a)
    report["inter_judge"] = {
        "cohen_kappa": stats.cohen_kappa([a[i] for i in ids], [b[i] for i in ids]),
        "raw_agreement": sum(a[i] == b[i] for i in ids) / len(ids),
    }
    # "Both judges agree it is supported" as a stricter combined judge.
    hum = {r["id"]: r["human"] for r in rows}
    both = [a[i] and b[i] for i in ids]
    report["combined_AND"] = {
        "balanced_accuracy": stats.balanced_accuracy([hum[i] for i in ids], both),
        "cohen_kappa_vs_human": stats.cohen_kappa([hum[i] for i in ids], both),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"summary": report, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main_async())
