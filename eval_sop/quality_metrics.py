"""
Quality metrics computed in code from the pipeline's actual data, to compare
with the numbers the Critic LLM reports about itself
(agents/critic_agent.py: CriticStructuredOutput.coverage_score,
source_diversity_score, contradiction_rate — all generated text).
"""

from __future__ import annotations

import math
from collections import Counter
from urllib.parse import urlparse


def domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def domain_diversity(claims: list[dict]) -> dict:
    """
    unique_ratio: distinct domains / distinct cited URLs (1.0 = every URL from a
        different site).
    entropy_norm: Shannon entropy of the domain distribution over all claim
        citations, divided by log(#citations) (0 = one site, 1 = all different).
    """
    cites = [u for c in claims for u in (c.get("source_urls") or []) if u]
    if not cites:
        return {"n_citations": 0, "unique_ratio": None, "entropy_norm": None, "n_domains": 0}
    doms = [domain(u) for u in cites]
    cnt = Counter(doms)
    n = len(doms)
    ent = -sum((k / n) * math.log(k / n) for k in cnt.values())
    return {
        "n_citations": n,
        "n_domains": len(cnt),
        "unique_ratio": len(cnt) / len(set(cites)),
        "entropy_norm": ent / math.log(n) if n > 1 else 0.0,
    }


def contradiction_rate(claims: list[dict]) -> float | None:
    """Fraction of claims with contradicting_sources > 0 (what the Critic is asked for)."""
    if not claims:
        return None
    return sum(1 for c in claims if (c.get("contradicting_sources") or 0) > 0) / len(claims)


def structural_coverage(sub_questions: list[str], claims: list[dict]) -> float | None:
    """Fraction of sub-questions with >=1 extracted claim. A floor, not semantic coverage."""
    if not sub_questions:
        return None
    have = {c.get("context") for c in claims}
    return sum(1 for q in sub_questions if q in have) / len(sub_questions)
