"""
Celery worker — runs the LangGraph research pipeline asynchronously.

The FastAPI endpoint submits a job to the Redis queue and returns immediately.
This worker picks it up and executes the full multi-agent pipeline.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import structlog
from celery import Celery

from config.logging import setup_logging

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Celery app — broker and result backend both use Redis
# ---------------------------------------------------------------------------
import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "research_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # re-queue on worker crash
    worker_prefetch_multiplier=1,  # one task at a time per worker
    task_soft_time_limit=900,  # 15 min soft limit
    task_time_limit=1200,  # 20 min hard limit
    task_max_retries=2,
    task_default_retry_delay=30,
)


@celery_app.task(
    bind=True,
    name="research_worker.submit_report_job",
    max_retries=2,
    default_retry_delay=30,
)
def submit_report_job(
    self,
    *,
    job_id: str,
    query: str,
    report_mode: str,
    max_rounds: int,
    token_budget: int,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-5",
    api_key: str | None = None,
):
    """
    Celery task: Execute the full research pipeline for a job.
    Uses asyncio.run() to run the async LangGraph pipeline synchronously.
    """
    setup_logging()

    log.info("worker_task_start", job_id=job_id, query=query[:80])
    t0 = time.perf_counter()

    # Import here to avoid circular imports at module load time
    from api.database import sync_update_job, sync_get_db
    from agents.graph import run_pipeline
    from agents.schemas import ReportMode, ResearchState

    try:
        # Mark job as running
        sync_update_job(
            job_id, {"status": "running", "celery_task_id": self.request.id}
        )

        # Build initial state. provider/model/api_key drive the BYOK LLM client;
        # api_key lives only in memory here and is never written to the DB.
        state = ResearchState(
            job_id=job_id,
            query=query,
            report_mode=ReportMode(report_mode),
            max_rounds=max_rounds,
            token_budget=token_budget,
            provider=provider,
            model=model,
            api_key=api_key,
        )

        # Run the async pipeline
        final_state = asyncio.run(run_pipeline(state))

        duration = time.perf_counter() - t0

        if final_state.fatal_error:
            raise RuntimeError(final_state.fatal_error)

        # Persist completed job
        sync_update_job(
            job_id,
            {
                "status": "completed",
                "report": final_state.final_report,
                "tokens_used": final_state.tokens_used_total,
                "cost_usd": final_state.cost_usd_total,
                "current_round": final_state.current_round,
                "activity_log": [
                    e.model_dump(mode="json") for e in final_state.activity_log
                ],
                "duration_seconds": duration,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        log.info(
            "worker_task_complete",
            job_id=job_id,
            duration=round(duration, 2),
            tokens=final_state.tokens_used_total,
            cost=round(final_state.cost_usd_total, 4),
        )

    except Exception as exc:
        duration = time.perf_counter() - t0
        log.error(
            "worker_task_failed",
            job_id=job_id,
            error=str(exc),
            duration=round(duration, 2),
        )

        sync_update_job(
            job_id,
            {
                "status": "failed",
                "error": str(exc)[:500],
                "duration_seconds": duration,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Retry on transient errors
        if "RateLimitError" in str(type(exc).__name__) or "ConnectionError" in str(
            type(exc).__name__
        ):
            raise self.retry(exc=exc, countdown=30)

        raise
