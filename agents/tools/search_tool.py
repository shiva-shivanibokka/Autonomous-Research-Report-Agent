"""
Search tool wrapping Tavily's advanced search API.
Returns ranked SearchResult objects with relevance scores.
"""

from __future__ import annotations

import time
from urllib.parse import urlparse

import structlog
from opentelemetry import trace
from tavily import AsyncTavilyClient

from agents.schemas import SearchResult

log = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


async def tavily_search(
    query: str,
    *,
    max_results: int = 8,
    search_depth: str = "advanced",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> list[SearchResult]:
    """
    Run a Tavily search and return ranked SearchResult objects.

    Args:
        query: The search query
        max_results: Number of results to return (max 10)
        search_depth: "basic" or "advanced" (advanced costs 2 API credits)
        include_domains: Only return results from these domains
        exclude_domains: Exclude these domains from results

    Returns:
        List of SearchResult sorted by relevance descending
    """
    with tracer.start_as_current_span("tool.tavily_search") as span:
        span.set_attribute("search.query", query)
        span.set_attribute("search.max_results", max_results)

        t0 = time.perf_counter()
        client = AsyncTavilyClient()

        kwargs: dict = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": False,
            "include_raw_content": False,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains

        response = await client.search(**kwargs)
        duration = time.perf_counter() - t0

        results: list[SearchResult] = []
        for r in response.get("results", []):
            domain = urlparse(r.get("url", "")).netloc
            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("content", ""),
                    relevance_score=r.get("score", 0.5),
                    published_date=r.get("published_date"),
                    source_domain=domain,
                )
            )

        # Sort by relevance descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        span.set_attribute("search.results_count", len(results))
        span.set_attribute("search.duration_seconds", duration)

        log.info(
            "tavily_search_complete",
            query=query[:100],
            results=len(results),
            duration_seconds=round(duration, 3),
        )

        return results
