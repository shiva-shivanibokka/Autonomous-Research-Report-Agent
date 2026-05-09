"""
Fact-Checker Agent — dynamically spawned for flagged claims.

For each claim the Critic flagged, runs a targeted Tavily search
and returns: Verified / Contradicted / Inconclusive.
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

import structlog
from pydantic import BaseModel

from agents.llm_client import FAST_MODEL, call_llm_structured
from agents.schemas import (
    AgentActivityEntry,
    AgentStatus,
    ClaimVerdict,
    ConfidenceLevel,
    FactCheckResult,
    FlaggedClaim,
    ResearchState,
)
from agents.tools.search_tool import tavily_search

log = structlog.get_logger(__name__)


class FactCheckStructuredOutput(BaseModel):
    verdict: str  # "verified" | "contradicted" | "inconclusive"
    evidence: str
    confidence: str  # "high" | "medium" | "low" | "inconclusive"
    explanation: str


async def fact_check_claim(
    flagged: FlaggedClaim,
    *,
    job_id: str,
    token_budget: int,
) -> FactCheckResult:
    """Run targeted search + LLM verification for one flagged claim."""
    t0 = time.perf_counter()
    log.info("fact_check_start", job_id=job_id, claim=flagged.claim_text[:80])

    # Targeted search for this specific claim
    try:
        search_results = await tavily_search(
            query=flagged.re_search_query,
            max_results=5,
            search_depth="advanced",
        )
    except Exception as e:
        log.error("fact_check_search_failed", error=str(e))
        return FactCheckResult(
            claim_text=flagged.claim_text,
            verdict=ClaimVerdict.INCONCLUSIVE,
            evidence=f"Search failed: {str(e)[:200]}",
            source_urls=[],
            confidence=ConfidenceLevel.INCONCLUSIVE,
        )

    if not search_results:
        return FactCheckResult(
            claim_text=flagged.claim_text,
            verdict=ClaimVerdict.INCONCLUSIVE,
            evidence="No search results found for verification query.",
            source_urls=[],
            confidence=ConfidenceLevel.INCONCLUSIVE,
        )

    # Build evidence text from search snippets
    evidence_text = "\n".join(
        f"Source {i + 1} ({r.source_domain}): {r.snippet}"
        for i, r in enumerate(search_results[:5])
    )
    source_urls = [r.url for r in search_results[:5]]

    system = """You are a fact-checker. Given a claim and search results, determine if the claim 
is verified, contradicted, or inconclusive based solely on the provided evidence.

- "verified": Multiple independent sources confirm the claim
- "contradicted": At least one credible source directly contradicts the claim  
- "inconclusive": Evidence is insufficient or ambiguous

Return ONLY valid JSON matching the schema."""

    user = f"""Claim to verify: {flagged.claim_text}

Original reason for flagging: {flagged.reason}

Search evidence:
{evidence_text}

Assess whether this claim is verified, contradicted, or inconclusive based on the evidence."""

    try:
        result, inp, out, cost = await call_llm_structured(
            system=system,
            user=user,
            output_schema=FactCheckStructuredOutput,
            model=FAST_MODEL,
            max_tokens=512,
            agent_name="fact_checker",
            token_budget_remaining=token_budget,
        )

        verdict_map = {
            "verified": ClaimVerdict.VERIFIED,
            "contradicted": ClaimVerdict.CONTRADICTED,
            "inconclusive": ClaimVerdict.INCONCLUSIVE,
        }
        conf_map = {
            "high": ConfidenceLevel.HIGH,
            "medium": ConfidenceLevel.MEDIUM,
            "low": ConfidenceLevel.LOW,
            "inconclusive": ConfidenceLevel.INCONCLUSIVE,
        }

        duration = time.perf_counter() - t0
        log.info(
            "fact_check_complete",
            job_id=job_id,
            claim=flagged.claim_text[:60],
            verdict=result.verdict,
            duration=round(duration, 3),
        )

        return FactCheckResult(
            claim_text=flagged.claim_text,
            verdict=verdict_map.get(result.verdict.lower(), ClaimVerdict.INCONCLUSIVE),
            evidence=result.evidence,
            source_urls=source_urls,
            confidence=conf_map.get(
                result.confidence.lower(), ConfidenceLevel.INCONCLUSIVE
            ),
            tokens_used=inp + out,
            duration_seconds=duration,
        )

    except Exception as e:
        duration = time.perf_counter() - t0
        log.error("fact_check_llm_failed", error=str(e))
        return FactCheckResult(
            claim_text=flagged.claim_text,
            verdict=ClaimVerdict.INCONCLUSIVE,
            evidence=f"Fact-check failed: {str(e)[:200]}",
            source_urls=source_urls,
            confidence=ConfidenceLevel.INCONCLUSIVE,
            duration_seconds=duration,
        )


async def run_fact_checking(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Dynamically spawn Fact-Checker agents for all flagged claims.
    Runs in parallel with bounded concurrency.
    """
    if not state.critic_output or not state.critic_output.flagged_claims:
        log.info("fact_checking_skipped_no_flags", job_id=state.job_id)
        state.activity_log.append(
            AgentActivityEntry(
                agent_name="Fact-Checker Agents",
                status=AgentStatus.SKIPPED,
                message="No claims flagged by Critic — skipping fact-checking.",
            )
        )
        state.converged = True
        return state

    flagged = state.critic_output.flagged_claims
    log.info("fact_checking_start", job_id=state.job_id, flagged_claims=len(flagged))

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Fact-Checker Agents",
            status=AgentStatus.RUNNING,
            message=f"Spawning {len(flagged)} fact-checker agents for flagged claims...",
        )
    )

    budget_per = (state.token_budget - state.tokens_used_total) // max(len(flagged), 1)

    tasks = [
        fact_check_claim(fc, job_id=state.job_id, token_budget=budget_per)
        for fc in flagged
    ]

    raw = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[FactCheckResult] = []
    total_tokens = 0

    for i, r in enumerate(raw):
        if isinstance(r, Exception):
            log.error("fact_checker_exception", error=str(r))
            results.append(
                FactCheckResult(
                    claim_text=flagged[i].claim_text,
                    verdict=ClaimVerdict.INCONCLUSIVE,
                    evidence=str(r)[:200],
                    source_urls=[],
                    confidence=ConfidenceLevel.INCONCLUSIVE,
                )
            )
        else:
            results.append(r)
            total_tokens += r.tokens_used

    state.fact_check_results = results
    state.tokens_used_total += total_tokens
    state.tokens_by_agent["fact_checker"] = (
        state.tokens_by_agent.get("fact_checker", 0) + total_tokens
    )

    verified = sum(1 for r in results if r.verdict == ClaimVerdict.VERIFIED)
    contradicted = sum(1 for r in results if r.verdict == ClaimVerdict.CONTRADICTED)
    inconclusive = sum(1 for r in results if r.verdict == ClaimVerdict.INCONCLUSIVE)

    state.converged = True  # After fact-checking, we always proceed to writing

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Fact-Checker Agents",
            status=AgentStatus.COMPLETED,
            message=(
                f"Fact-check complete: {verified} verified, "
                f"{contradicted} contradicted, {inconclusive} inconclusive"
            ),
            tokens_used=total_tokens,
        )
    )

    log.info(
        "fact_checking_complete",
        job_id=state.job_id,
        verified=verified,
        contradicted=contradicted,
        inconclusive=inconclusive,
        tokens=total_tokens,
    )

    return state
