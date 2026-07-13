"""
In-process job runner — the Cloud-Run-friendly alternative to the Celery worker.

When JOB_BACKEND=inline (the default), the API runs the LangGraph pipeline as an
asyncio task in its own event loop instead of dispatching to a separate Celery
worker + Redis. This lets the whole backend deploy as a single scale-to-zero
Cloud Run service, and it uses the shared async DB pool (no connection-per-write).

Trade-off (ponytail): a job is tied to the API instance running it. While the UI
polls, the instance stays warm and the job progresses; if the instance is
reclaimed mid-job the job is lost. Fine for an interactive demo — set
JOB_BACKEND=celery (docker-compose) for the durable, separately-scaled path.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import structlog

log = structlog.get_logger(__name__)


async def run_job_inline(
    *,
    job_id: str,
    query: str,
    report_mode: str,
    max_rounds: int,
    token_budget: int,
    provider: str,
    model: str,
    api_key: str | None,
) -> None:
    """Run the full research pipeline in-process and persist the result."""
    from agents.graph import run_pipeline
    from agents.schemas import ReportMode, ResearchState
    from api.database import update_job_status

    t0 = time.perf_counter()
    log.info("inline_job_start", job_id=job_id, provider=provider, model=model)

    try:
        await update_job_status(job_id, {"status": "running"})

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

        final_state = await run_pipeline(state)
        duration = time.perf_counter() - t0

        if final_state.fatal_error:
            raise RuntimeError(final_state.fatal_error)

        await update_job_status(
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
            "inline_job_complete",
            job_id=job_id,
            duration=round(duration, 2),
            tokens=final_state.tokens_used_total,
        )

    except Exception as exc:
        duration = time.perf_counter() - t0
        log.error("inline_job_failed", job_id=job_id, error=str(exc))
        await update_job_status(
            job_id,
            {
                "status": "failed",
                "error": str(exc)[:500],
                "duration_seconds": duration,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
