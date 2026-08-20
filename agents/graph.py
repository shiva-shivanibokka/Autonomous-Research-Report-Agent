"""
LangGraph pipeline graph — wires all agents into a stateful, self-improving research pipeline.

Graph topology:
  orchestrate
      ↓
  parallel_search
      ↓
  parallel_scraping
      ↓
  parallel_analysis
      ↓
  critic_review
      ↓ (conditional)
  ┌── [needs_more_research=True AND rounds < max_rounds] ──→ increment_round → orchestrate (loop)
  └── [converged] ──→ fact_checking
                            ↓
                       writer_agent
                            ↓
                          END
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog
from langgraph.graph import END, StateGraph

from agents.analyst_agent import run_parallel_analysis
from agents.critic_agent import run_critic_agent, should_continue_research
from agents.fact_checker_agent import run_fact_checking
from agents.llm_client import set_llm_creds
from agents.orchestrator import orchestrate
from agents.schemas import AgentActivityEntry, AgentStatus, ResearchState
from agents.scraper_agent import run_parallel_scraping
from agents.search_agent import run_parallel_search
from agents.writer_agent import run_writer_agent

log = structlog.get_logger(__name__)


async def increment_round(state: ResearchState) -> ResearchState:
    """
    Transition node: increment the round counter before looping back to orchestrate.
    Clears intermediate outputs so the new round starts fresh (but keeps approved claims).
    """
    state.current_round += 1

    # Preserve approved claims from the Critic for context, clear raw pipeline outputs
    state.search_outputs = []
    state.scraper_outputs = []
    state.analyst_outputs = []
    # Note: critic_output is preserved — orchestrator reads it to refine sub-questions

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Pipeline",
            status=AgentStatus.RUNNING,
            message=f"Starting re-research round {state.current_round + 1} of {state.max_rounds}...",
        )
    )

    log.info(
        "pipeline_round_increment", job_id=state.job_id, new_round=state.current_round
    )
    return state


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph research pipeline.

    Returns the compiled graph ready for .ainvoke().
    """
    # Use dict-based state (LangGraph requirement for StateGraph)
    # We serialize/deserialize ResearchState at the entry/exit points

    graph = StateGraph(dict)

    # Register all nodes
    graph.add_node("orchestrate", _wrap(orchestrate))
    graph.add_node("parallel_search", _wrap(run_parallel_search))
    graph.add_node("parallel_scraping", _wrap(run_parallel_scraping))
    graph.add_node("parallel_analysis", _wrap(run_parallel_analysis))
    graph.add_node("critic_review", _wrap(run_critic_agent))
    graph.add_node("increment_round", _wrap(increment_round))
    graph.add_node("fact_checking", _wrap(run_fact_checking))
    graph.add_node("writer", _wrap(run_writer_agent))

    # Set entry point
    graph.set_entry_point("orchestrate")

    # Linear edges: orchestrate → search → scrape → analyze → critic
    graph.add_edge("orchestrate", "parallel_search")
    graph.add_edge("parallel_search", "parallel_scraping")
    graph.add_edge("parallel_scraping", "parallel_analysis")
    graph.add_edge("parallel_analysis", "critic_review")

    # Conditional edge from critic: re-research loop OR proceed to fact-checking
    graph.add_conditional_edges(
        "critic_review",
        _should_continue_research,
        {
            "re_research": "increment_round",
            "fact_check": "fact_checking",
        },
    )

    # Loop back edge: increment_round → orchestrate (for re-research)
    graph.add_edge("increment_round", "orchestrate")

    # Terminal path: fact_checking → writer → END
    graph.add_edge("fact_checking", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


def _wrap(agent_fn):
    """
    Wrap an agent function that takes/returns ResearchState to work with
    LangGraph's dict-based state.
    """

    async def wrapper(state_dict: dict) -> dict:
        state = ResearchState(**state_dict)
        updated = await agent_fn(state)
        return updated.model_dump()

    return wrapper


def _should_continue_research(state_dict: dict) -> str:
    """Conditional edge wrapper — deserializes state and delegates to critic logic."""
    state = ResearchState(**state_dict)
    return should_continue_research(state)


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_pipeline(
    state: ResearchState,
    on_progress: Callable[[ResearchState], Awaitable[None]] | None = None,
) -> ResearchState:
    """
    Entry point for running the full research pipeline.

    Args:
        state: Initial ResearchState with query and config
        on_progress: awaited with the full state after each graph node. This is
            what makes the activity feed live — the pipeline runs for minutes and
            the UI polls every few seconds, so without it the feed stays empty
            for the whole run and then fills in at once at the end.

    Returns:
        Final ResearchState with completed report
    """
    log.info(
        "pipeline_start",
        job_id=state.job_id,
        query=state.query[:80],
        mode=state.report_mode.value,
        provider=state.provider.value,
        model=state.model,
        max_rounds=state.max_rounds,
        token_budget=state.token_budget,
    )

    # Set the BYOK provider/model/key for this job. Every agent's call_llm() in
    # this async context reads these creds. api_key stays in memory only.
    set_llm_creds(state.provider.value, state.model, state.api_key)

    graph = get_graph()
    latest: dict | None = None

    try:
        # stream_mode="values" yields the whole state after every node, which is
        # both the progress hook and — because `latest` survives an exception —
        # the only way to keep the activity log of a run that died partway. The
        # previous ainvoke() returned nothing on failure, so a failed job showed
        # an empty feed and the user had no idea which agent broke.
        async for chunk in graph.astream(state.model_dump(), stream_mode="values"):
            latest = chunk
            # A node that sets fatal_error has already recorded why in the
            # activity log. Stop here rather than running the remaining agents
            # against empty inputs and spending more LLM calls to reach the
            # same conclusion.
            if chunk.get("fatal_error"):
                log.warning(
                    "pipeline_halted_by_node",
                    job_id=state.job_id,
                    error=str(chunk["fatal_error"])[:200],
                )
                break
            if on_progress is not None:
                try:
                    await on_progress(ResearchState(**chunk))
                except Exception as progress_error:
                    # Progress reporting must never be able to kill the research.
                    log.warning(
                        "pipeline_progress_callback_failed",
                        job_id=state.job_id,
                        error=str(progress_error),
                    )

        final_state = ResearchState(**latest) if latest else state

        log.info(
            "pipeline_halted" if final_state.fatal_error else "pipeline_complete",
            job_id=final_state.job_id,
            rounds=final_state.current_round + 1,
            total_tokens=final_state.tokens_used_total,
            total_cost=round(final_state.cost_usd_total, 4),
            converged=final_state.converged,
            has_report=final_state.final_report is not None,
        )

        return final_state

    except Exception as e:
        log.error("pipeline_fatal_error", job_id=state.job_id, error=str(e))
        partial = ResearchState(**latest) if latest else state
        partial.fatal_error = str(e)
        partial.errors.append(str(e))
        return partial
