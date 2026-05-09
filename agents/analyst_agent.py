"""
Analyst Agent — one per sub-question, runs in parallel.

Reads scraped content, extracts key claims, detects contradictions,
and assigns confidence levels based on source triangulation.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from pydantic import BaseModel

from agents.llm_client import FAST_MODEL, call_llm_structured
from agents.schemas import (
    AgentActivityEntry,
    AgentStatus,
    AnalystAgentOutput,
    Claim,
    ConfidenceLevel,
    ResearchState,
    ScraperAgentOutput,
)

log = structlog.get_logger(__name__)


class RawClaimList(BaseModel):
    claims: list[dict]  # raw before Pydantic validation
    contradictions_found: int
    analysis_summary: str


def _assign_confidence(supporting: int, contradicting: int) -> ConfidenceLevel:
    """Source triangulation: confidence based on independent source agreement."""
    if contradicting > 0:
        return ConfidenceLevel.CONTESTED
    if supporting >= 3:
        return ConfidenceLevel.HIGH
    if supporting == 2:
        return ConfidenceLevel.MEDIUM
    if supporting == 1:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.INCONCLUSIVE


async def run_analyst_agent(
    sub_question: str,
    scraper_output: ScraperAgentOutput,
    *,
    job_id: str,
    token_budget: int,
) -> AnalystAgentOutput:
    """
    Analyze scraped content for a single sub-question.
    Extracts claims, detects contradictions, scores confidence.
    """
    t0 = time.perf_counter()
    log.info("analyst_agent_start", job_id=job_id, sub_question=sub_question[:60])

    successful_pages = [
        p for p in scraper_output.pages if p.content and not p.scrape_error
    ]

    if not successful_pages:
        log.warning("analyst_no_content", job_id=job_id, sub_question=sub_question[:60])
        return AnalystAgentOutput(
            sub_question=sub_question,
            key_claims=[],
            contradictions_found=0,
            avg_confidence=0.0,
            error="No content available to analyze",
        )

    # Build content block for the LLM
    content_block = ""
    for i, page in enumerate(successful_pages, 1):
        content_block += f"\n--- SOURCE {i}: {page.title[:80]} ---\nURL: {page.url}\n{page.content[:3000]}\n"

    system = """You are a research analyst. Your job is to extract key claims from web content 
and assess how many independent sources support each claim.

For each claim:
1. State the claim clearly and concisely
2. List which source URLs support it  
3. Count supporting_sources (how many of the provided sources agree)
4. Count contradicting_sources (how many sources contradict it)
5. Note any contradiction_detail if sources disagree
6. Extract any specific data_point (numbers, statistics, dates) if present

Return ONLY valid JSON matching the schema. Be precise. Do not fabricate claims not present in the sources."""

    user = f"""Sub-question being analyzed: {sub_question}

Scraped content from {len(successful_pages)} sources:
{content_block}

Extract the most important factual claims that answer the sub-question.
Return 4-8 key claims maximum. Focus on claims with the strongest evidence."""

    class AnalystStructuredOutput(BaseModel):
        claims: list[dict]
        contradictions_found: int
        avg_confidence_score: float

    try:
        result, inp, out, cost = await call_llm_structured(
            system=system,
            user=user,
            output_schema=AnalystStructuredOutput,
            model=FAST_MODEL,
            max_tokens=2048,
            agent_name="analyst",
            token_budget_remaining=token_budget,
        )
    except Exception as e:
        duration = time.perf_counter() - t0
        log.error("analyst_llm_failed", sub_question=sub_question[:60], error=str(e))
        return AnalystAgentOutput(
            sub_question=sub_question,
            key_claims=[],
            contradictions_found=0,
            avg_confidence=0.0,
            duration_seconds=duration,
            error=str(e)[:300],
            tokens_used=0,
        )

    # Build validated Claim objects with source triangulation
    claims: list[Claim] = []
    for raw in result.claims:
        try:
            supporting = int(raw.get("supporting_sources", 1))
            contradicting = int(raw.get("contradicting_sources", 0))
            confidence = _assign_confidence(supporting, contradicting)

            claims.append(
                Claim(
                    text=str(raw.get("text", raw.get("claim", ""))),
                    source_urls=raw.get("source_urls", []),
                    confidence=confidence,
                    supporting_sources=supporting,
                    contradicting_sources=contradicting,
                    data_point=raw.get("data_point"),
                    contradiction_detail=raw.get("contradiction_detail"),
                )
            )
        except Exception:
            continue

    # Compute average confidence score
    conf_scores = {
        ConfidenceLevel.HIGH: 1.0,
        ConfidenceLevel.MEDIUM: 0.67,
        ConfidenceLevel.LOW: 0.33,
        ConfidenceLevel.CONTESTED: 0.2,
        ConfidenceLevel.INCONCLUSIVE: 0.0,
    }
    avg_conf = (
        sum(conf_scores[c.confidence] for c in claims) / len(claims) if claims else 0.0
    )

    duration = time.perf_counter() - t0
    log.info(
        "analyst_agent_complete",
        job_id=job_id,
        sub_question=sub_question[:60],
        claims=len(claims),
        contradictions=result.contradictions_found,
        avg_confidence=round(avg_conf, 3),
        tokens=inp + out,
        duration=round(duration, 3),
    )

    return AnalystAgentOutput(
        sub_question=sub_question,
        key_claims=claims,
        contradictions_found=result.contradictions_found,
        avg_confidence=avg_conf,
        tokens_used=inp + out,
        duration_seconds=duration,
    )


async def run_parallel_analysis(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Fan out Analyst Agents in parallel across all sub-questions.
    """
    log.info("parallel_analysis_start", job_id=state.job_id)

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Analyst Agents",
            status=AgentStatus.RUNNING,
            message=f"Analyzing content across {len(state.scraper_outputs)} sub-questions in parallel...",
        )
    )

    budget_remaining = state.token_budget - state.tokens_used_total
    budget_per_agent = budget_remaining // max(len(state.scraper_outputs), 1)

    tasks = [
        run_analyst_agent(
            scraper_out.sub_question,
            scraper_out,
            job_id=state.job_id,
            token_budget=budget_per_agent,
        )
        for scraper_out in state.scraper_outputs
    ]

    raw = await asyncio.gather(*tasks, return_exceptions=True)

    analyst_outputs: list[AnalystAgentOutput] = []
    total_tokens = 0
    total_claims = 0

    for i, result in enumerate(raw):
        if isinstance(result, Exception):
            log.error("analyst_agent_exception", error=str(result))
            analyst_outputs.append(
                AnalystAgentOutput(
                    sub_question=state.scraper_outputs[i].sub_question,
                    key_claims=[],
                    contradictions_found=0,
                    avg_confidence=0.0,
                    error=str(result)[:300],
                )
            )
        else:
            analyst_outputs.append(result)
            total_tokens += result.tokens_used
            total_claims += len(result.key_claims)

    state.analyst_outputs = analyst_outputs
    state.tokens_used_total += total_tokens
    state.tokens_by_agent["analyst"] = (
        state.tokens_by_agent.get("analyst", 0) + total_tokens
    )

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Analyst Agents",
            status=AgentStatus.COMPLETED,
            message=f"Extracted {total_claims} claims across {len(analyst_outputs)} sub-questions",
            tokens_used=total_tokens,
        )
    )

    return state
