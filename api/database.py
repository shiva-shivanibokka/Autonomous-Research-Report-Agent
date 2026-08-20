"""
Job store. Postgres (asyncpg) when DATABASE_URL is set, in-process otherwise.

Stores job metadata, status, activity logs, and completed reports.

**On the in-process fallback:** with the default JOB_BACKEND=inline the pipeline
runs as an asyncio task inside the same process that serves the status polls, so
a dict is not an approximation of the database there — it is the same
single-writer, single-reader store without the network hop, the container, or
the account. That is the condition that makes it safe, and it is the same
trade-off inline mode already documents: a job is tied to the instance running
it and does not survive a restart.

It exists so `pip install -r requirements.txt && uvicorn api.main:app` works with
no infrastructure at all. Set DATABASE_URL for durable, multi-instance storage —
docker-compose and any real deployment do. /health reports which one answered, so
the substitution is never silent.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import asyncpg
import structlog

log = structlog.get_logger(__name__)

# Empty means "no database configured here", not "use the compose hostname".
# The old default was postgres://…@postgres:5432/…, which resolves only inside
# docker-compose; everywhere else startup blocked on a DNS lookup that could
# never succeed, to learn something already knowable from the environment.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

_pool: asyncpg.Pool | None = None

# In-process fallback store: job_id -> row dict, insertion-ordered.
_memory_jobs: dict[str, dict[str, Any]] = {}


def backend_name() -> str:
    """Which store is actually answering — surfaced on /health."""
    return "postgres" if DATABASE_URL else "memory"


async def init_db():
    """Create connection pool and initialize schema (no-op for the memory store)."""
    global _pool
    if not DATABASE_URL:
        log.warning(
            "database_not_configured_using_memory_store",
            detail="Jobs are held in process memory and lost on restart. "
            "Set DATABASE_URL for durable storage.",
        )
        return

    _pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    await _create_schema()
    log.info("database_connected", url=DATABASE_URL.split("@")[-1])


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("database_closed")


async def _create_schema():
    """Create tables if they don't exist."""
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS research_jobs (
                job_id          VARCHAR(36) PRIMARY KEY,
                status          VARCHAR(20) NOT NULL DEFAULT 'queued',
                query           TEXT NOT NULL,
                report_mode     VARCHAR(40) NOT NULL DEFAULT 'general',
                max_rounds      INTEGER NOT NULL DEFAULT 2,
                token_budget    INTEGER NOT NULL DEFAULT 80000,
                celery_task_id  VARCHAR(36),
                current_round   INTEGER DEFAULT 0,
                tokens_used     INTEGER DEFAULT 0,
                cost_usd        NUMERIC(10, 6) DEFAULT 0,
                duration_seconds NUMERIC(8, 2),
                activity_log    JSONB DEFAULT '[]',
                report          JSONB,
                error           TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at    TIMESTAMPTZ
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status ON research_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON research_jobs(created_at DESC);
        """)
    log.info("database_schema_initialized")


async def create_job(
    query: str,
    report_mode: str,
    max_rounds: int,
    token_budget: int,
) -> str:
    """Insert a new job and return its job_id."""
    import uuid

    job_id = str(uuid.uuid4())

    if not DATABASE_URL:
        _memory_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "query": query,
            "report_mode": report_mode,
            "max_rounds": max_rounds,
            "token_budget": token_budget,
            "current_round": 0,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "duration_seconds": None,
            "activity_log": [],
            "report": None,
            "error": None,
            "created_at": datetime.now(UTC),
            "completed_at": None,
        }
        log.info("job_created", job_id=job_id, store="memory")
        return job_id

    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO research_jobs (job_id, query, report_mode, max_rounds, token_budget, status)
            VALUES ($1, $2, $3, $4, $5, 'queued')
            """,
            job_id,
            query,
            report_mode,
            max_rounds,
            token_budget,
        )

    log.info("job_created", job_id=job_id)
    return job_id


async def get_job(job_id: str) -> dict | None:
    """Fetch a job by ID."""
    if not DATABASE_URL:
        job = _memory_jobs.get(job_id)
        return dict(job) if job else None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM research_jobs WHERE job_id = $1",
            job_id,
        )

    if not row:
        return None

    result = dict(row)
    # asyncpg returns JSONB as dicts already
    return result


# Columns update_job_status is allowed to write. The SET clause is built by
# string interpolation (column names cannot be bound as parameters), so this is
# the only thing standing between a future caller passing a user-controlled key
# and arbitrary SQL. Values are always bound.
_UPDATABLE_COLUMNS = frozenset(
    {
        "status",
        "celery_task_id",
        "current_round",
        "tokens_used",
        "cost_usd",
        "duration_seconds",
        "activity_log",
        "report",
        "error",
        "completed_at",
    }
)

# TIMESTAMPTZ columns. asyncpg binds these through its datetime codec, which
# raises TypeError on a str rather than parsing it — so an ISO string here is a
# hard failure, not a slow path.
_TIMESTAMP_COLUMNS = frozenset({"completed_at"})


def _prepare_updates(updates: dict[str, Any]) -> tuple[list[str], list[Any]]:
    """
    Validate column names and coerce values into the types asyncpg will accept.

    Both callers used to hand `completed_at` an ISO-8601 *string*, which asyncpg
    rejects outright (pgproto/codecs/datetime.pyx: "expected a datetime.date or
    datetime.datetime instance"). That made the final write of every successful
    job throw — and the except-handler that tried to record the failure wrote
    completed_at the same way, so it threw too and the job was left on 'running'
    forever. Coercing here fixes both call sites and any future one.
    """
    set_clauses: list[str] = []
    values: list[Any] = []

    for i, (key, value) in enumerate(updates.items(), start=1):
        if key not in _UPDATABLE_COLUMNS:
            raise ValueError(f"refusing to update unknown column {key!r}")
        set_clauses.append(f"{key} = ${i}")

        if key in _TIMESTAMP_COLUMNS and isinstance(value, str):
            value = datetime.fromisoformat(value)
        elif isinstance(value, (dict, list)):
            value = json.dumps(value)
        values.append(value)

    return set_clauses, values


async def update_job_status(job_id: str, updates: dict[str, Any]):
    """Update job fields by job_id."""
    if not updates:
        return

    if not DATABASE_URL:
        job = _memory_jobs.get(job_id)
        if job is None:
            return
        for key, value in updates.items():
            if key not in _UPDATABLE_COLUMNS:
                raise ValueError(f"refusing to update unknown column {key!r}")
            if key in _TIMESTAMP_COLUMNS and isinstance(value, str):
                value = datetime.fromisoformat(value)
            job[key] = value
        return

    set_clauses, values = _prepare_updates(updates)
    values.append(job_id)
    set_clause = ", ".join(set_clauses)

    async with _pool.acquire() as conn:
        await conn.execute(
            f"UPDATE research_jobs SET {set_clause} WHERE job_id = ${len(values)}",  # noqa: S608 — column names are checked against _UPDATABLE_COLUMNS above; all values are bound
            *values,
        )


SUMMARY_FIELDS = (
    "job_id",
    "status",
    "query",
    "report_mode",
    "current_round",
    "tokens_used",
    "cost_usd",
    "duration_seconds",
    "created_at",
    "completed_at",
    "error",
)


async def get_recent_jobs(limit: int = 10, offset: int = 0) -> list[dict]:
    """List recent jobs with summary fields only."""
    if not DATABASE_URL:
        newest_first = sorted(
            _memory_jobs.values(), key=lambda j: j["created_at"], reverse=True
        )
        return [
            {k: job.get(k) for k in SUMMARY_FIELDS}
            for job in newest_first[offset : offset + limit]
        ]

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT job_id, status, query, report_mode, current_round,
                   tokens_used, cost_usd, duration_seconds, created_at, completed_at, error
            FROM research_jobs
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Sync versions for Celery worker (which can't use async)
# ---------------------------------------------------------------------------


def sync_update_job(job_id: str, updates: dict[str, Any]):
    """Synchronous job update for use inside Celery tasks."""
    import asyncio

    if not DATABASE_URL:
        # A Celery worker is a separate process, so the API's in-memory store is
        # unreachable from here — the write would silently vanish. Celery mode
        # requires a real database; say so instead of no-op'ing (ledger rule 6:
        # reads may degrade, writes fail loudly).
        raise RuntimeError(
            "JOB_BACKEND=celery requires DATABASE_URL — the in-process job store "
            "is not shared across processes."
        )

    asyncio.run(_async_update_for_sync(job_id, updates))


async def _async_update_for_sync(job_id: str, updates: dict[str, Any]):
    """One-off connection for the Celery worker, which has no shared pool."""
    set_clauses, values = _prepare_updates(updates)
    values.append(job_id)
    set_clause = ", ".join(set_clauses)

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute(
            f"UPDATE research_jobs SET {set_clause} WHERE job_id = ${len(values)}",  # noqa: S608 — column names are checked against _UPDATABLE_COLUMNS above; all values are bound
            *values,
        )
    finally:
        await conn.close()
