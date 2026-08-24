"""
Search tool wrapping the Tavily search API.
Returns ranked SearchResult objects with relevance scores.
"""

from __future__ import annotations

import os
import time
from urllib.parse import urlparse

import structlog
from opentelemetry import trace
from tavily import AsyncTavilyClient

from agents.schemas import SearchResult

log = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# One client for the process. A fresh AsyncTavilyClient per search builds a new
# httpx connection pool and never closes it — at 5 sub-questions x 3 rounds plus
# fact-checks that is dozens of leaked pools per job.
_client: AsyncTavilyClient | None = None


def _get_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        if not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError(
                "TAVILY_API_KEY is not set — web search is required for the "
                "research pipeline. Add it to .env (see .env.example)."
            )
        _client = AsyncTavilyClient()
    return _client


# Tavily bills "advanced" at 2 credits per search and "basic" at 1. A two-round
# run issues 10-15 searches, so the choice is the difference between roughly 26
# and 13 credits per report — which matters a great deal on a free tier and not
# at all on a paid one. Default to basic and let anyone who wants the deeper
# crawl opt in, rather than silently spending double.
DEFAULT_SEARCH_DEPTH = (
    "advanced"
    if os.environ.get("TAVILY_SEARCH_DEPTH", "").strip().lower() == "advanced"
    else "basic"
)


def _clamp_score(score: object) -> float:
    """Tavily relevance -> the 0.0-1.0 the schema requires. None means unscored."""
    try:
        return min(1.0, max(0.0, float(score)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5


async def tavily_search(
    query: str,
    *,
    max_results: int = 8,
    search_depth: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> list[SearchResult]:
    """
    Run a Tavily search and return ranked SearchResult objects.

    Args:
        query: The search query
        max_results: Number of results to return (max 10)
        search_depth: "basic" (1 credit) or "advanced" (2). Defaults to
            TAVILY_SEARCH_DEPTH, else "basic".
        include_domains: Only return results from these domains
        exclude_domains: Exclude these domains from results

    Returns:
        List of SearchResult sorted by relevance descending
    """
    with tracer.start_as_current_span("tool.tavily_search") as span:
        span.set_attribute("search.query", query)
        span.set_attribute("search.max_results", max_results)

        search_depth = search_depth or DEFAULT_SEARCH_DEPTH
        t0 = time.perf_counter()
        client = _get_client()

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
                    url=r.get("url") or "",
                    title=r.get("title") or "",
                    snippet=r.get("content") or "",
                    relevance_score=_clamp_score(r.get("score")),
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
            depth=search_depth,
            duration_seconds=round(duration, 3),
        )

        return results
