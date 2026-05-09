"""
Critic Agent — the quality gate before the Writer.

Reviews ALL Analyst findings:
- Flags unsupported or low-confidence claims
- Detects coverage gaps
- Scores source diversity
- Decides whether another research round is needed
- Generates targeted re-search queries for flagged claims
"""

from __future__ import annotations

import time

import structlog
from pydantic import BaseModel

from agents.llm_client import REASON_MODEL, call_llm_structured
from agents.schemas import (
    AgentActivityEntry,
    AgentStatus,
    Claim,
    ConfidenceLevel,
    CriticAgentOutput,
    FlaggedClaim,
    ResearchState,
)

log = structlog.get_logger(__name__)

# Thresholds
COVERAGE_THRESHOLD = 0.70  # below this → needs more research
QUALITY_THRESHOLD = 0.65  # below this → flag for re-research
MAX_FLAGGED_PER_ROUND = 5  # cap how many claims we re-research per round


class CriticStructuredOutput(BaseModel):
    flagged_claims: list[dict]
    overall_quality_score: float
    coverage_score: float
    source_diversity_score: float
    contradiction_rate: float
    needs_more_research: bool
    critic_notes: str


async def run_critic_agent(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Review all analyst findings and decide if more research is needed.
    This node is the trigger point for the self-improving loop.
    """
    log.info("critic_agent_start", job_id=state.job_id, round=state.current_round)

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Critic Agent",
            status=AgentStatus.RUNNING,
            message="Reviewing all analyst findings for quality and coverage...",
        )
    )

    # Aggregate all claims across sub-questions
    all_claims: list[Claim] = []
    for analyst_out in state.analyst_outputs:
        all_claims.extend(analyst_out.key_claims)

    if not all_claims:
        log.warning("critic_no_claims", job_id=state.job_id)
        critic_output = CriticAgentOutput(
            flagged_claims=[],
            approved_claims=[],
            overall_quality_score=0.0,
            coverage_score=0.0,
            source_diversity_score=0.0,
            contradiction_rate=0.0,
            needs_more_research=False,
            critic_notes="No claims found to review — research pipeline produced no output.",
        )
        state.critic_output = critic_output
        return state

    # Build claims summary for the LLM
    claims_text = ""
    for i, claim in enumerate(all_claims, 1):
        claims_text += (
            f"\n[{i}] CLAIM: {claim.text}\n"
            f"    Sources: {len(claim.source_urls)} | Confidence: {claim.confidence.value} | "
            f"Supporting: {claim.supporting_sources} | Contradicting: {claim.contradicting_sources}\n"
        )
        if claim.contradiction_detail:
            claims_text += f"    Contradiction: {claim.contradiction_detail}\n"

    sub_questions_text = "\n".join(f"- {q}" for q in state.sub_questions)

    system = """You are a senior research critic. Your job is to review research findings and 
determine whether they meet publication quality standards.

For each flagged claim, provide:
- The exact claim text
- The reason it is flagged (unsupported, single source, contradicted, vague)
- A targeted re-search query to find better evidence

Scoring guidelines:
- coverage_score: What fraction of the sub-questions are well-answered? (0.0-1.0)
- source_diversity_score: How varied are the source domains? High = many different domains. (0.0-1.0)  
- overall_quality_score: Weighted average of coverage and source diversity (0.0-1.0)
- contradiction_rate: What fraction of claims have contradictions? (0.0-1.0)
- needs_more_research: true if coverage_score < 0.70 OR overall_quality_score < 0.65

Return ONLY valid JSON matching the schema."""

    user = f"""Research query: {state.query}
Report mode: {state.report_mode.value}
Research round: {state.current_round + 1} of {state.max_rounds}

Sub-questions that should be covered:
{sub_questions_text}

All extracted claims ({len(all_claims)} total):
{claims_text}

Review these findings and:
1. Flag claims that are LOW confidence, CONTESTED, or INCONCLUSIVE (max {MAX_FLAGGED_PER_ROUND} flags)
2. Score the overall research quality
3. Decide if another research round is warranted"""

    t0 = time.perf_counter()
    try:
        result, inp, out, cost = await call_llm_structured(
            system=system,
            user=user,
            output_schema=CriticStructuredOutput,
            model=REASON_MODEL,
            max_tokens=2048,
            agent_name="critic",
            token_budget_remaining=state.token_budget - state.tokens_used_total,
        )
    except Exception as e:
        log.error("critic_llm_failed", error=str(e))
        # Fail open — approve everything, don't block the pipeline
        state.critic_output = CriticAgentOutput(
            flagged_claims=[],
            approved_claims=all_claims,
            overall_quality_score=0.6,
            coverage_score=0.6,
            source_diversity_score=0.6,
            contradiction_rate=0.0,
            needs_more_research=False,
            critic_notes=f"Critic failed: {str(e)[:200]}. Proceeding with available research.",
        )
        return state

    duration = time.perf_counter() - t0

    # Parse flagged claims
    flagged: list[FlaggedClaim] = []
    flagged_texts = set()
    for raw in result.flagged_claims[:MAX_FLAGGED_PER_ROUND]:
        try:
            fc = FlaggedClaim(
                claim_text=str(raw.get("claim_text", raw.get("text", ""))),
                reason=str(raw.get("reason", "")),
                sub_question=str(
                    raw.get(
                        "sub_question",
                        state.sub_questions[0] if state.sub_questions else "",
                    )
                ),
                re_search_query=str(
                    raw.get("re_search_query", raw.get("suggested_search", ""))
                ),
            )
            if fc.claim_text not in flagged_texts:
                flagged.append(fc)
                flagged_texts.add(fc.claim_text)
        except Exception:
            continue

    # Approved = all claims minus those that appear in flagged set
    approved = [c for c in all_claims if c.text not in flagged_texts]

    # Enforce max rounds — don't loop forever
    needs_more = (
        result.needs_more_research
        and state.current_round < state.max_rounds - 1
        and len(flagged) > 0
    )

    critic_output = CriticAgentOutput(
        flagged_claims=flagged,
        approved_claims=approved,
        overall_quality_score=result.overall_quality_score,
        coverage_score=result.coverage_score,
        source_diversity_score=result.source_diversity_score,
        contradiction_rate=result.contradiction_rate,
        needs_more_research=needs_more,
        critic_notes=result.critic_notes,
        tokens_used=inp + out,
        duration_seconds=duration,
    )

    state.critic_output = critic_output
    state.tokens_used_total += inp + out
    state.cost_usd_total += cost
    state.tokens_by_agent["critic"] = state.tokens_by_agent.get("critic", 0) + inp + out

    status_msg = (
        f"Quality score: {result.overall_quality_score:.2f} | "
        f"Coverage: {result.coverage_score:.2f} | "
        f"{len(flagged)} claims flagged"
    )
    if needs_more:
        status_msg += f" → Triggering re-research round {state.current_round + 2}"

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Critic Agent",
            status=AgentStatus.COMPLETED,
            message=status_msg,
            tokens_used=inp + out,
            cost_usd=cost,
        )
    )

    log.info(
        "critic_agent_complete",
        job_id=state.job_id,
        quality=result.overall_quality_score,
        coverage=result.coverage_score,
        flagged=len(flagged),
        needs_more_research=needs_more,
        tokens=inp + out,
    )

    return state


def should_continue_research(state: ResearchState) -> str:
    """
    LangGraph conditional edge: route to re-research or proceed to fact-checking.
    Returns "re_research" or "fact_check"
    """
    if (
        state.critic_output
        and state.critic_output.needs_more_research
        and state.current_round < state.max_rounds - 1
    ):
        return "re_research"
    return "fact_check"
