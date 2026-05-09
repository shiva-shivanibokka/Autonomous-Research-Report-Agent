"""
PostgreSQL async database layer using asyncpg.
Stores job metadata, status, activity logs, and completed reports.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import asyncpg
import structlog

log = structlog.get_logger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://research_user:research_pass@postgres:5432/research_db",
)

_pool: asyncpg.Pool | None = None


async def init_db():
    """Create connection pool and initialize schema."""
    global _pool
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


async def update_job_status(job_id: str, updates: dict[str, Any]):
    """Update job fields by job_id."""
    if not updates:
        return

    # Build SET clause dynamically
    set_clauses = []
    values = []
    for i, (key, value) in enumerate(updates.items(), start=1):
        set_clauses.append(f"{key} = ${i}")
        if isinstance(value, (dict, list)):
            values.append(json.dumps(value))
        else:
            values.append(value)

    values.append(job_id)
    set_clause = ", ".join(set_clauses)

    async with _pool.acquire() as conn:
        await conn.execute(
            f"UPDATE research_jobs SET {set_clause} WHERE job_id = ${len(values)}",
            *values,
        )


async def get_recent_jobs(limit: int = 10, offset: int = 0) -> list[dict]:
    """List recent jobs with summary fields only."""
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

    asyncio.run(_async_update_for_sync(job_id, updates))


async def _async_update_for_sync(job_id: str, updates: dict[str, Any]):
    """Create a temporary pool for sync context."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        set_clauses = []
        values = []
        for i, (key, value) in enumerate(updates.items(), start=1):
            set_clauses.append(f"{key} = ${i}")
            if isinstance(value, (dict, list)):
                values.append(json.dumps(value))
            else:
                values.append(value)

        values.append(job_id)
        set_clause = ", ".join(set_clauses)
        await conn.execute(
            f"UPDATE research_jobs SET {set_clause} WHERE job_id = ${len(values)}",
            *values,
        )
    finally:
        await conn.close()


def sync_get_db():
    """Placeholder — sync context uses direct asyncpg connect, not pool."""
    return None
