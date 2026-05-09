"""
Job management helpers — thin wrappers around database.py for the API layer.
"""

from __future__ import annotations

from typing import Any

from api.database import create_job as _create_job
from api.database import get_job as _get_job
from api.database import get_recent_jobs as _get_recent_jobs
from api.database import update_job_status as _update_job_status


async def create_job(
    query: str,
    report_mode: str,
    max_rounds: int,
    token_budget: int,
) -> str:
    return await _create_job(
        query=query,
        report_mode=report_mode,
        max_rounds=max_rounds,
        token_budget=token_budget,
    )


async def get_job(job_id: str) -> dict | None:
    return await _get_job(job_id)


async def get_recent_jobs(limit: int = 10, offset: int = 0) -> list[dict]:
    return await _get_recent_jobs(limit=limit, offset=offset)


async def update_job_status(job_id: str, updates: dict[str, Any]):
    return await _update_job_status(job_id, updates)
