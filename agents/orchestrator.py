"""
Orchestrator Agent — the brain of the pipeline.

Responsibilities:
- Decompose the research query into 3-5 focused sub-questions
- Assign token budgets to parallel agent branches
- Decide if more research is needed after Critic review
- Spawn additional Fact-Checker targets when Critic flags claims
- Track convergence across self-improving rounds
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel

from agents.llm_client import REASON_MODEL, call_llm_structured
from agents.schemas import AgentActivityEntry, AgentStatus, ResearchState

log = structlog.get_logger(__name__)


class SubQuestionList(BaseModel):
    sub_questions: list[str]
    reasoning: str  # why these sub-questions cover the query


async def orchestrate(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Decompose query into sub-questions and allocate token budgets.
    Runs at the start of every research round.
    """
    log.info(
        "orchestrator_start",
        job_id=state.job_id,
        round=state.current_round,
        query=state.query[:80],
    )

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Orchestrator",
            status=AgentStatus.RUNNING,
            message=f"Round {state.current_round + 1}: decomposing query into sub-questions...",
        )
    )

    # On re-research rounds, incorporate Critic feedback
    critic_context = ""
    if state.critic_output and state.current_round > 0:
        flagged = state.critic_output.flagged_claims
        if flagged:
            critic_context = (
                "\n\nThe previous research round was reviewed by a Critic agent. "
                "The following claims were flagged as needing more evidence:\n"
            )
            for fc in flagged[:5]:
                critic_context += f"- {fc.claim_text[:120]} (suggested search: {fc.re_search_query})\n"
            critic_context += (
                "\nPlease include sub-questions that specifically address these gaps."
            )

    system = f"""You are a research decomposition expert. Your job is to break down a complex research 
query into {3 if state.current_round == 0 else 2}-5 focused, non-overlapping sub-questions that together 
fully answer the main query.

Report mode: {state.report_mode.value}

Rules:
- Each sub-question must be independently searchable on the web
- Sub-questions should cover different aspects (not overlapping)
- For competitive_intelligence: include market overview, key players, differentiators, trends
- For investment_thesis: include market size, growth drivers, risks, comparables
- For academic_literature_review: include foundational papers, recent advances, methodology gaps
- For general: cover breadth of the topic evenly
- Be specific enough that a web search returns relevant results{critic_context}"""

    user = f"Research query: {state.query}"

    budget_per_call = state.token_budget // 20  # ~5% of budget for orchestration
    try:
        result, inp, out, cost = await call_llm_structured(
            system=system,
            user=user,
            output_schema=SubQuestionList,
            model=REASON_MODEL,
            max_tokens=min(1024, budget_per_call),
            agent_name="orchestrator",
            token_budget_remaining=state.token_budget - state.tokens_used_total,
        )
    except Exception as e:
        # Every other LLM-calling agent handles its own failures; this one did
        # not, and it runs first — so the commonest failure of all (a rejected
        # API key) escaped as a raw exception before a single activity entry was
        # recorded. LangGraph discards a node's mutations when it raises, so the
        # user got an empty feed and a 401 with no indication of which agent
        # produced it. Record it, mark the run fatal, and let the graph unwind.
        log.error("orchestrator_llm_failed", job_id=state.job_id, error=str(e))
        state.activity_log.append(
            AgentActivityEntry(
                agent_name="Orchestrator",
                status=AgentStatus.FAILED,
                message=f"Could not decompose the query: {str(e)[:200]}",
            )
        )
        state.errors.append(f"Orchestrator failed: {e}")
        state.fatal_error = f"Orchestrator failed: {e}"
        return state

    # On re-research rounds, add targeted sub-questions from Critic flags
    if state.current_round > 0 and state.critic_output:
        for fc in state.critic_output.flagged_claims[:3]:
            targeted_q = fc.re_search_query.strip()
            if targeted_q and targeted_q not in result.sub_questions:
                result.sub_questions.append(targeted_q)

    state.sub_questions = result.sub_questions
    state.tokens_used_total += inp + out
    state.cost_usd_total += cost
    state.tokens_by_agent["orchestrator"] = (
        state.tokens_by_agent.get("orchestrator", 0) + inp + out
    )

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Orchestrator",
            status=AgentStatus.COMPLETED,
            message=f"Decomposed into {len(result.sub_questions)} sub-questions: {'; '.join(q[:40] + '...' for q in result.sub_questions)}",
            tokens_used=inp + out,
            cost_usd=cost,
        )
    )

    log.info(
        "orchestrator_complete",
        job_id=state.job_id,
        sub_questions=len(result.sub_questions),
        tokens=inp + out,
        cost=cost,
    )

    return state
