"""
Prometheus metrics for the research pipeline.
Exposes all metrics at /metrics for Prometheus scraping.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Request / job counters
# ---------------------------------------------------------------------------

REPORT_REQUESTS = Counter(
    "research_report_requests_total",
    "Total number of report generation requests",
    ["mode"],  # competitive_intelligence, investment_thesis, etc.
)

REPORT_COMPLETIONS = Counter(
    "research_report_completions_total",
    "Total number of successfully completed reports",
    ["mode"],
)

REPORT_ERRORS = Counter(
    "research_report_errors_total",
    "Total number of failed report generations",
)

# ---------------------------------------------------------------------------
# Duration histograms
# ---------------------------------------------------------------------------

REPORT_DURATION = Histogram(
    "research_report_duration_seconds",
    "End-to-end report generation duration in seconds",
    ["mode"],
    buckets=[30, 60, 120, 180, 300, 480, 600, 900, 1200],
)

AGENT_DURATION = Histogram(
    "research_agent_duration_seconds",
    "Individual agent execution duration in seconds",
    ["agent_name"],
    buckets=[1, 2, 5, 10, 20, 30, 60, 120],
)

LLM_CALL_DURATION = Histogram(
    "research_llm_call_duration_seconds",
    "LLM API call duration in seconds",
    ["model", "agent"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
)

# ---------------------------------------------------------------------------
# Token and cost counters
# ---------------------------------------------------------------------------

LLM_TOKENS_TOTAL = Counter(
    "research_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "type"],  # type: input/output
)

LLM_COST_USD = Counter(
    "research_llm_cost_usd_total",
    "Total LLM cost in USD",
    ["model"],
)

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

REPORT_QUALITY_SCORE = Histogram(
    "research_report_quality_score",
    "Distribution of report quality scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

CRITIC_REJECTIONS = Counter(
    "research_critic_rejections_total",
    "Number of claims flagged/rejected by the Critic agent",
)

FACT_CHECK_VERDICTS = Counter(
    "research_fact_check_verdicts_total",
    "Fact-checker verdicts",
    ["verdict"],  # verified, contradicted, inconclusive
)

RESEARCH_ROUNDS = Histogram(
    "research_pipeline_rounds",
    "Number of self-improving research rounds per report",
    buckets=[1, 2, 3],
)

# ---------------------------------------------------------------------------
# Scraping metrics
# ---------------------------------------------------------------------------

PAGES_SCRAPED = Counter(
    "research_pages_scraped_total",
    "Total web pages scraped",
    ["method"],  # httpx, playwright
)

SCRAPE_ERRORS = Counter(
    "research_scrape_errors_total",
    "Total scraping failures",
)

# ---------------------------------------------------------------------------
# System gauges
# ---------------------------------------------------------------------------

ACTIVE_JOBS = Gauge(
    "research_active_jobs",
    "Number of currently running research jobs",
)

QUEUE_DEPTH = Gauge(
    "research_queue_depth",
    "Number of jobs waiting in the Celery queue",
)


def track_active_jobs():
    """Increment active job counter when a new job starts."""
    ACTIVE_JOBS.inc()


def release_active_job():
    """Decrement active job counter when a job completes."""
    ACTIVE_JOBS.dec()
