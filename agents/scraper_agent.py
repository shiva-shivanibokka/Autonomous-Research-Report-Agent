"""
Scraper Agent — scrapes full content from top URLs per sub-question.
Handles JS-rendered sites via Playwright fallback.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from agents.schemas import (
    AgentActivityEntry,
    AgentStatus,
    ResearchState,
    ScraperAgentOutput,
)
from agents.tools.scraper_tool import scrape_pages

log = structlog.get_logger(__name__)

# How many top URLs to scrape per sub-question
TOP_N_URLS = 4


async def run_scraper_agent(
    sub_question: str,
    urls_with_titles: list[tuple[str, str]],
    *,
    job_id: str,
) -> ScraperAgentOutput:
    """Scrape top URLs for a single sub-question."""
    t0 = time.perf_counter()
    log.info(
        "scraper_agent_start",
        job_id=job_id,
        sub_question=sub_question[:60],
        urls=len(urls_with_titles),
    )

    pages = await scrape_pages(urls_with_titles, max_concurrent=3)
    duration = time.perf_counter() - t0

    # Filter out failed scrapes (empty content)
    successful = [p for p in pages if p.content and not p.scrape_error]
    log.info(
        "scraper_agent_complete",
        job_id=job_id,
        sub_question=sub_question[:60],
        scraped=len(successful),
        failed=len(pages) - len(successful),
        duration=round(duration, 3),
    )

    return ScraperAgentOutput(
        sub_question=sub_question,
        pages=pages,
        duration_seconds=duration,
    )


async def run_parallel_scraping(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Scrape top URLs for every sub-question in parallel.
    Pairs each search output with its top-N URLs and scrapes concurrently.
    """
    log.info("parallel_scraping_start", job_id=state.job_id)

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Scraper Agents",
            status=AgentStatus.RUNNING,
            message=f"Scraping top {TOP_N_URLS} sources per sub-question...",
        )
    )

    tasks = []
    for search_out in state.search_outputs:
        top_results = search_out.results[:TOP_N_URLS]
        urls_with_titles = [(r.url, r.title) for r in top_results]
        tasks.append(
            run_scraper_agent(
                search_out.sub_question,
                urls_with_titles,
                job_id=state.job_id,
            )
        )

    raw = await asyncio.gather(*tasks, return_exceptions=True)

    scraper_outputs: list[ScraperAgentOutput] = []
    for i, result in enumerate(raw):
        if isinstance(result, Exception):
            log.error("scraper_agent_exception", error=str(result))
            scraper_outputs.append(
                ScraperAgentOutput(
                    sub_question=state.search_outputs[i].sub_question,
                    pages=[],
                )
            )
        else:
            scraper_outputs.append(result)

    state.scraper_outputs = scraper_outputs
    total_pages = sum(len(o.pages) for o in scraper_outputs)
    successful_pages = sum(
        sum(1 for p in o.pages if p.content and not p.scrape_error)
        for o in scraper_outputs
    )
    playwright_pages = sum(
        sum(1 for p in o.pages if p.used_playwright) for o in scraper_outputs
    )

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Scraper Agents",
            status=AgentStatus.COMPLETED,
            message=(
                f"Scraped {successful_pages}/{total_pages} pages successfully"
                + (f" ({playwright_pages} via Playwright)" if playwright_pages else "")
            ),
        )
    )

    return state
