"""
Audit the repo's recorded showcase run (frontend/public/demo/run.json) — the
one real end-to-end pipeline run that exists (claude-sonnet-4-5, 2 rounds).

  python -m eval_sop.showcase_audit fetch     # repo venv: fetch cited pages with the pipeline's scraper
  python -m eval_sop.showcase_audit judge     # torch env: NLI support per (sentence, cited URL)
  python -m eval_sop.showcase_audit analyze   # either env: metrics -> results/showcase_audit.json

What is measured, all from the recorded data, none of it from the Critic:
* citation-level support: every report sentence that carries an inline
  "(https://...)" citation is checked against the page at that URL, fetched
  now with the pipeline's own scraper (same 8,000-char cleaning cap);
* domain diversity of the listed citations (30; the Writer truncates the list)
  and of the inline citations;
* internal consistency of the reported quality block (contradiction_rate vs
  the contested count, flagged vs fact-checked).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "frontend" / "public" / "demo" / "run.json"
PAGES = HERE / "cache" / "showcase_pages.json"
RAW = HERE / "results" / "showcase_nli_raw.jsonl"
OUT = HERE / "results" / "showcase_audit.json"
# "(https://a)" or "(https://a, https://b)"
URL_IN_PARENS = re.compile(r"\((https?://[^\s),]+(?:,\s*https?://[^\s),]+)*)\)")
URL = re.compile(r"https?://[^\s),]+")
SENT = re.compile(r"[^.!?]+(?:[.!?]+(?=\s|$)|$)")


def report_texts(rep: dict) -> list[tuple[str, str]]:
    out = [("executive_summary", rep.get("executive_summary", ""))]
    out += [(f"key_finding_{i}", t) for i, t in enumerate(rep.get("key_findings", []))]
    out += [(f"section:{k}", v) for k, v in rep.get("detailed_sections", {}).items()]
    return out


def citation_pairs() -> list[dict]:
    rep = json.loads(RUN.read_text(encoding="utf-8"))["report"]
    pairs = []
    for where, text in report_texts(rep):
        for para in text.split("\n"):
            # Sentence boundaries are found on the text with URLs masked, so the
            # dots inside URLs don't split sentences.
            masked = URL_IN_PARENS.sub(lambda m: "(" + "U" * (len(m.group(0)) - 2) + ")", para)
            for m in SENT.finditer(masked):
                seg = para[m.start() : m.end()]
                urls = [u for g in URL_IN_PARENS.findall(seg) for u in URL.findall(g)]
                if not urls:
                    continue
                claim = " ".join(URL_IN_PARENS.sub("", seg).split()).strip()
                if len(claim) < 20:
                    continue
                pairs.append({"where": where, "claim": claim, "urls": urls})
    return pairs


async def fetch():
    sys.path.insert(0, str(HERE.parent))
    from agents.tools.scraper_tool import scrape_page

    urls = sorted({u for p in citation_pairs() for u in p["urls"]})
    have = json.loads(PAGES.read_text(encoding="utf-8")) if PAGES.exists() else {}
    for u in urls:
        if u in have:
            continue
        page = await scrape_page(u, "")
        have[u] = {"content": page.content, "error": page.scrape_error}
        PAGES.parent.mkdir(exist_ok=True)
        PAGES.write_text(json.dumps(have), encoding="utf-8")
        print(("ok  " if page.content else "FAIL"), u, flush=True)


def judge():
    from eval_sop.nli import MODELS, NLIJudge

    pages = json.loads(PAGES.read_text(encoding="utf-8"))
    pairs = citation_pairs()
    RAW.parent.mkdir(exist_ok=True)
    done = set()
    if RAW.exists():
        done = {(j["judge"], j["claim"], j["url"]) for j in map(json.loads, open(RAW, encoding="utf-8"))}
    with open(RAW, "a", encoding="utf-8") as f:
        for name in MODELS:
            j = NLIJudge(name)
            for p in pairs:
                for u in p["urls"]:
                    if (name, p["claim"], u) in done:
                        continue
                    content = (pages.get(u) or {}).get("content") or ""
                    if content.strip():
                        s = j.score(content, p["claim"])
                    else:
                        s = {"p_entail": None, "supported": None, "n_windows": 0}
                    f.write(json.dumps({"judge": name, "claim": p["claim"], "url": u,
                                        "page_missing": not content.strip(), **s}) + "\n")
                    f.flush()
            del j
            print("judged", name, flush=True)


def analyze():
    from collections import Counter

    from eval_sop import stats
    from eval_sop.quality_metrics import domain

    run = json.loads(RUN.read_text(encoding="utf-8"))
    rep = run["report"]
    q = rep["quality"]
    pairs = citation_pairs()
    raw = [json.loads(line) for line in open(RAW, encoding="utf-8")] if RAW.exists() else []
    out = {"run": {k: run[k] for k in ("query", "provider", "model", "rounds_run", "max_rounds",
                                        "converged", "tokens_used", "cost_usd", "duration_seconds")}}
    out["reported_quality_block"] = q

    # Diversity
    cites = rep["citations"]
    doms = [domain(c["url"]) for c in cites]
    inline = [u for p in pairs for u in p["urls"]]
    out["diversity"] = {
        "critic_self_reported_source_diversity": q["source_diversity_score"],
        "sources_consulted_per_quality_block": q["total_sources_consulted"],
        "listed_citations_in_report": len(cites),  # writer_agent.py caps the list at 30
        "listed_unique_domains": len(set(doms)),
        "listed_unique_domain_ratio": len(set(doms)) / len(cites),
        "listed_top_domains": Counter(doms).most_common(5),
        "inline_citations": len(inline),
        "inline_unique_urls": len(set(inline)),
        "inline_unique_domains": len({domain(u) for u in inline}),
        "inline_unique_domain_ratio": len({domain(u) for u in inline}) / max(len(set(inline)), 1),
        "inline_cited_urls_missing_from_listed_citations": sorted(set(inline) - {c["url"] for c in cites}),
    }
    cd = q["confidence_distribution"]
    out["consistency"] = {
        "critic_contradiction_rate": q["contradiction_rate"],
        "contested_claims_in_distribution": cd["contested"],
        "computed_contradiction_rate_from_distribution": cd["contested"] / max(sum(cd.values()), 1),
        "contradictions_mapped_in_report": len(rep.get("contradictions", [])),
        "claims_flagged_by_critic_final_round": q["claims_flagged_by_critic"],
        "claims_verified_by_fact_checker": q["claims_verified_by_fact_checker"],
        "round1_flags_in_activity_log": [e for e in run["activity_log"] if "flagged" in e.get("message", "")]
        if isinstance(run["activity_log"][0], dict) else None,
    }

    # Citation support
    from eval_sop.nli import MODELS

    sup = {}
    for name in MODELS:
        rows = [r for r in raw if r["judge"] == name]
        by_claim = {}
        for r in rows:
            by_claim.setdefault(r["claim"], []).append(r)
        claim_ok, pair_ok = [], []
        for p in pairs:
            rs = by_claim.get(p["claim"], [])
            judged = [r for r in rs if not r["page_missing"]]
            pair_ok += [bool(r["supported"]) for r in judged]
            if judged:
                claim_ok.append(any(r["supported"] for r in judged))
        sup[name] = {
            "n_cited_sentences": len(pairs),
            "n_sentences_with_fetchable_page": len(claim_ok),
            "sentence_supported_by_any_cited_page": stats.mean_ci([float(x) for x in claim_ok], boot=5000),
            "n_sentence_url_pairs_judged": len(pair_ok),
            "pair_supported": stats.mean_ci([float(x) for x in pair_ok], boot=5000),
            "pairs_page_missing": sum(1 for r in rows if r["page_missing"]),
        }
    out["citation_support_nli"] = sup
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str)[:5000])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        asyncio.run(fetch())
    elif cmd == "judge":
        judge()
    else:
        analyze()
