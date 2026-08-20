"""
Search Agent — runs in parallel, one instance per sub-question.

Uses Tavily advanced search and returns ranked SearchResult objects
with relevance scores.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from agents.schemas import (
    AgentActivityEntry,
    AgentStatus,
    ResearchState,
    SearchAgentOutput,
)
from agents.tools.search_tool import tavily_search

log = structlog.get_logger(__name__)


async def run_search_agent(
    sub_question: str,
    *,
    job_id: str,
    token_budget: int = 0,  # not consumed (Tavily doesn't use tokens)
    max_results: int = 8,
) -> SearchAgentOutput:
    """
    Run one Search Agent for a single sub-question.
    Returns a SearchAgentOutput with ranked results.
    """
    log.info("search_agent_start", job_id=job_id, sub_question=sub_question[:80])
    t0 = time.perf_counter()

    try:
        results = await tavily_search(
            query=sub_question,
            max_results=max_results,
            search_depth="advanced",
        )
        duration = time.perf_counter() - t0

        log.info(
            "search_agent_complete",
            job_id=job_id,
            sub_question=sub_question[:60],
            results=len(results),
            duration=round(duration, 3),
        )

        return SearchAgentOutput(
            sub_question=sub_question,
            results=results,
            duration_seconds=duration,
        )

    except Exception as e:
        duration = time.perf_counter() - t0
        log.error(
            "search_agent_failed",
            job_id=job_id,
            sub_question=sub_question[:60],
            error=str(e),
        )
        return SearchAgentOutput(
            sub_question=sub_question,
            results=[],
            duration_seconds=duration,
            error=str(e)[:300],
        )


async def run_parallel_search(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Fan out Search Agents in parallel across all sub-questions.
    Uses asyncio.gather with partial failure tolerance.
    """
    # Drop blanks before anything counts them: an empty sub-question reaches
    # Tavily as an empty query and also skews the per-agent token budget.
    # Seen live when the Critic emitted a flagged claim with no search text.
    state.sub_questions = [q.strip() for q in state.sub_questions if q and q.strip()]

    log.info(
        "parallel_search_start",
        job_id=state.job_id,
        sub_questions=len(state.sub_questions),
    )

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Search Agents",
            status=AgentStatus.RUNNING,
            message=f"Launching {len(state.sub_questions)} parallel search agents...",
        )
    )

    # Allocate token budget per sub-question (search doesn't consume tokens but track for accounting)
    budget_per_agent = (state.token_budget - state.tokens_used_total) // max(
        len(state.sub_questions), 1
    )

    tasks = [
        run_search_agent(
            q,
            job_id=state.job_id,
            token_budget=budget_per_agent,
            max_results=8,
        )
        for q in state.sub_questions
    ]

    # gather with return_exceptions=True so one failing search doesn't kill the whole pipeline
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    search_outputs: list[SearchAgentOutput] = []
    failed = 0
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            failed += 1
            log.error(
                "search_agent_exception",
                sub_question=state.sub_questions[i],
                error=str(result),
            )
            search_outputs.append(
                SearchAgentOutput(
                    sub_question=state.sub_questions[i],
                    results=[],
                    error=str(result)[:300],
                )
            )
        else:
            search_outputs.append(result)

    total_results = sum(len(o.results) for o in search_outputs)
    state.search_outputs = search_outputs

    # Accumulate across rounds so citations survive the re-research loop.
    for out in search_outputs:
        for r in out.results:
            state.all_sources.setdefault(r.url, r)

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Search Agents",
            status=AgentStatus.COMPLETED,
            message=(
                f"Retrieved {total_results} sources across {len(search_outputs)} sub-questions"
                + (f" ({failed} searches failed)" if failed else "")
            ),
        )
    )

    log.info(
        "parallel_search_complete",
        job_id=state.job_id,
        total_results=total_results,
        failed_agents=failed,
    )

    return state
