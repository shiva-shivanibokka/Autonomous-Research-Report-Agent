"""
FastAPI backend for the Autonomous Research Report Agent.

Endpoints:
  POST /report/generate     → Submit a new research job (returns job_id immediately)
  GET  /report/status/{id}  → Poll job status + live agent activity feed
  GET  /report/{id}         → Retrieve completed report
  GET  /reports             → List recent reports
  GET  /health              → Health check
  GET  /metrics             → Prometheus metrics
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import Response

from api.database import close_db, init_db
from api.jobs import (
    create_job,
    get_job,
    get_recent_jobs,
    update_job_status,
)
from api.metrics import (
    REPORT_DURATION,
    REPORT_ERRORS,
    REPORT_REQUESTS,
    track_active_jobs,
)
from api.worker import submit_report_job
from agents.llm_client import list_models
from agents.report_format import render_report_markdown
from agents.schemas import (
    JobStatusResponse,
    ReportMode,
    ReportRequest,
    ReportJobResponse,
    ReportResponse,
)
from config.logging import setup_logging
from config.otel import setup_tracing

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# Job execution backend: "inline" runs the pipeline in-process (single Cloud Run
# service, no Redis/Celery); "celery" dispatches to the worker (docker-compose).
JOB_BACKEND = os.environ.get("JOB_BACKEND", "inline").lower()

# Strong refs to in-flight inline tasks so the event loop doesn't GC them.
_inline_tasks: set = set()


# ---------------------------------------------------------------------------
# App lifespan — startup/shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    setup_tracing(service_name="research-api")
    await init_db()
    log.info("research_api_started")
    yield
    await close_db()
    log.info("research_api_stopped")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Autonomous Research Report Agent",
    description=(
        "Multi-agent research pipeline that autonomously produces "
        "cited, quality-scored research reports. "
        "Features: parallel agent execution, self-improving critic loop, "
        "source triangulation, Playwright scraping, per-report cost accounting."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Lock CORS to explicit origins (comma-separated ALLOWED_ORIGINS env var).
# Wildcard "*" with allow_credentials=True is rejected by browsers and unsafe;
# BYOK keys are sent from the browser, so the origin allowlist matters.
_allowed_origins = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:7860"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — structured request logging + trace ID injection
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time, uuid

    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    with structlog.contextvars.bound_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    ):
        response = await call_next(request)
        duration = time.perf_counter() - t0
        log.info(
            "http_request",
            status=response.status_code,
            duration_ms=round(duration * 1000, 1),
        )
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
async def health():
    return {
        "status": "ok",
        "service": "research-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------
@app.get("/metrics", tags=["ops"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# BYOK — list a provider's models using the caller's key (populates the UI dropdown)
# ---------------------------------------------------------------------------
@app.get("/providers/{provider}/models", tags=["providers"])
async def get_provider_models(provider: str, x_provider_key: str = Header(...)):
    """
    Return the live model list for a provider, fetched with the user's own key.
    The key is used only for this request and is never stored.
    """
    try:
        models = await list_models(provider, x_provider_key)
    except Exception as exc:
        # Bad key or unknown provider — tell the client without leaking internals.
        raise HTTPException(
            status_code=400,
            detail=f"Could not list models for provider '{provider}'. Check the provider and API key.",
        ) from exc
    return {"provider": provider, "models": models}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
@app.post("/report/generate", response_model=ReportJobResponse, tags=["reports"])
@limiter.limit("10/minute")
async def generate_report(request: Request, body: ReportRequest):
    """
    Submit a new research job. Returns immediately with a job_id.
    The actual pipeline runs asynchronously in a Celery worker.
    """
    REPORT_REQUESTS.labels(mode=body.report_mode.value).inc()

    job_id = await create_job(
        query=body.query,
        report_mode=body.report_mode.value,
        max_rounds=body.max_rounds,
        token_budget=body.token_budget,
    )

    # Dispatch the job. BYOK provider/model/key travel with it but the key is
    # never persisted — only in memory (inline) or transiently in the broker (celery).
    job_kwargs = dict(
        job_id=job_id,
        query=body.query,
        report_mode=body.report_mode.value,
        max_rounds=body.max_rounds,
        token_budget=body.token_budget,
        provider=body.provider.value,
        model=body.model,
        api_key=body.api_key,
    )
    if JOB_BACKEND == "celery":
        submit_report_job.delay(**job_kwargs)
    else:
        import asyncio

        from api.inline_runner import run_job_inline

        task = asyncio.create_task(run_job_inline(**job_kwargs))
        _inline_tasks.add(task)
        task.add_done_callback(_inline_tasks.discard)

    track_active_jobs()
    log.info(
        "report_job_submitted",
        job_id=job_id,
        query=body.query[:80],
        mode=body.report_mode.value,
    )

    return ReportJobResponse(
        job_id=job_id,
        status="queued",
        message="Research job queued. Poll /report/status/{job_id} for live updates.",
        created_at=datetime.now(timezone.utc),
    )


@app.get("/report/status/{job_id}", response_model=JobStatusResponse, tags=["reports"])
async def get_report_status(job_id: str):
    """
    Get live status of a research job, including the agent activity feed.
    Poll this endpoint every 2-3 seconds to show progress in the UI.
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        query=job["query"],
        report_mode=job["report_mode"],
        current_round=job.get("current_round", 0),
        tokens_used=job.get("tokens_used", 0),
        cost_usd=job.get("cost_usd", 0.0),
        activity_log=job.get("activity_log", []),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error"),
    )


@app.get("/report/{job_id}", response_model=ReportResponse, tags=["reports"])
async def get_report(job_id: str):
    """Retrieve the completed research report."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] == "running" or job["status"] == "queued":
        raise HTTPException(
            status_code=202,
            detail="Report is still being generated. Check /report/status/{job_id}.",
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {job.get('error', 'unknown error')}",
        )

    if not job.get("report"):
        raise HTTPException(status_code=404, detail="Report data not found")

    return ReportResponse(
        job_id=job["job_id"],
        query=job["query"],
        report_mode=job["report_mode"],
        report=job["report"],
        report_markdown=render_report_markdown(job["report"], job["report_mode"]),
        quality=job["report"].get("quality", {}),
        tokens_used=job.get("tokens_used", 0),
        cost_usd=job.get("cost_usd", 0.0),
        duration_seconds=job.get("duration_seconds", 0.0),
        created_at=job["created_at"],
        completed_at=job["completed_at"],
    )


@app.get("/reports", tags=["reports"])
async def list_reports(limit: int = 10, offset: int = 0):
    """List recent research reports with summary metadata."""
    jobs = await get_recent_jobs(limit=limit, offset=offset)
    return {"reports": jobs, "total": len(jobs)}


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    REPORT_ERRORS.inc()
    # Log the detail server-side; never return exception internals to the client.
    request_id = request.headers.get("X-Request-ID")
    log.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        request_id=request_id,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )
