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


def _as_count(*candidates: object, fallback: int = 0) -> int:
    """
    First candidate that can be read as a count. A list counts as its length.

    The Analyst prompt asks the model both to *list* the supporting source URLs
    and to *count* them, and models reasonably answer by putting a list of URLs
    in `supporting_sources` and the number in `supporting_sources_count`. The
    old code did int(raw["supporting_sources"]) and got a list, which raised
    inside a bare `except Exception: continue` — so every claim was silently
    dropped and the pipeline produced empty reports while still paying for the
    tokens. Accept whichever shape arrives.
    """
    for value in candidates:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            return len(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except ValueError:
                continue
    return fallback


def _as_url_list(*candidates: object) -> list[str]:
    """First candidate that looks like a list of source URLs."""
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, (list, tuple)):
            urls = [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
            if urls:
                return urls
    return []


def _coerce_claim(raw: dict) -> Claim | None:
    """
    Turn one raw LLM claim dict into a validated Claim, or None if unusable.

    Field names vary between models and between runs, so every field is read
    through a list of accepted spellings rather than one exact key.
    """
    if not isinstance(raw, dict):
        return None

    text = str(
        raw.get("text") or raw.get("claim") or raw.get("claim_text") or ""
    ).strip()
    if not text:
        return None

    urls = _as_url_list(
        raw.get("source_urls"),
        raw.get("sources"),
        raw.get("supporting_urls"),
        raw.get("supporting_sources"),
    )
    supporting = _as_count(
        raw.get("supporting_sources_count"),
        raw.get("num_supporting_sources"),
        raw.get("supporting_sources"),
        fallback=len(urls) or 1,
    )
    contradicting = _as_count(
        raw.get("contradicting_sources_count"),
        raw.get("num_contradicting_sources"),
        raw.get("contradicting_sources"),
        fallback=0,
    )

    detail = raw.get("contradiction_detail") or raw.get("contradiction")
    data_point = raw.get("data_point") or raw.get("data")

    return Claim(
        text=text,
        source_urls=urls,
        confidence=_assign_confidence(supporting, contradicting),
        supporting_sources=supporting,
        contradicting_sources=contradicting,
        data_point=str(data_point) if data_point else None,
        contradiction_detail=str(detail) if detail else None,
    )


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

    # Field names and types are spelled out because they used to be described in
    # prose ("List which source URLs support it" / "Count supporting_sources"),
    # which reads as two instructions for one key — and models answered with a
    # list of URLs where an integer was expected.
    system = """You are a research analyst. Your job is to extract key claims from web content
and assess how many independent sources support each claim.

Each entry in "claims" must be an object with exactly these keys:
  "text"                  : string  — the claim, stated clearly and concisely
  "source_urls"           : array of strings — the source URLs backing the claim
  "supporting_sources"    : integer — HOW MANY of the provided sources agree (a number, not a list)
  "contradicting_sources" : integer — HOW MANY of the provided sources contradict it (a number, not a list)
  "contradiction_detail"  : string or null — what the disagreement is, if any
  "data_point"            : string or null — any specific number, statistic or date in the claim

"supporting_sources" and "contradicting_sources" are counts. Put the URLs in
"source_urls" and the counts in those two fields — never a list in a count field.

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
            claim = _coerce_claim(raw)
        except Exception as exc:
            log.warning("analyst_claim_discarded", error=str(exc), raw=str(raw)[:200])
            continue
        if claim is None:
            log.warning("analyst_claim_unusable", raw=str(raw)[:200])
            continue
        claims.append(claim)

    if result.claims and not claims:
        # Extracting nothing from a non-empty model response means the shape
        # changed, not that the sources were empty. Loud, because this exact
        # failure ran silently and made every report contentless.
        log.error(
            "analyst_all_claims_discarded",
            sub_question=sub_question[:80],
            received=len(result.claims),
            sample=str(result.claims[0])[:300],
        )

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
        cost_usd=cost,
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
    total_cost = 0.0
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
            total_cost += result.cost_usd
            total_claims += len(result.key_claims)

    state.analyst_outputs = analyst_outputs
    state.tokens_used_total += total_tokens
    # The Analysts fan out one call per sub-question, so dropping their cost
    # understated every report's total — and by the largest share of the calls.
    state.cost_usd_total += total_cost
    state.tokens_by_agent["analyst"] = (
        state.tokens_by_agent.get("analyst", 0) + total_tokens
    )

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Analyst Agents",
            status=AgentStatus.COMPLETED,
            message=f"Extracted {total_claims} claims across {len(analyst_outputs)} sub-questions",
            tokens_used=total_tokens,
            cost_usd=total_cost,
        )
    )

    return state
