"""
API tests driven through FastAPI's TestClient.

These used to require a separately started uvicorn, a live Postgres, a Redis and
real provider keys, which meant they never ran — not in CI, not locally, and not
on the one path where the interesting bugs were. Everything here runs in-process
against the in-memory job store with no credentials, so it runs everywhere.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    # No DATABASE_URL -> the in-process job store. Set before importing the app
    # so api.database picks it up at module load.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")

    import api.database as database

    monkeypatch.setattr(database, "DATABASE_URL", "", raising=False)
    database._memory_jobs.clear()

    from api.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Ops endpoints
# ---------------------------------------------------------------------------
def test_health_reports_which_job_store_answered(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    # A fallback that reports the name of the thing it replaced hides outages.
    assert body["job_store"] == "memory"
    assert body["job_backend"] == "inline"


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"research_report_requests_total" in resp.content


# ---------------------------------------------------------------------------
# Listing — bounded pagination
# ---------------------------------------------------------------------------
def test_list_reports_empty(client):
    body = client.get("/reports").json()
    assert body["reports"] == []


@pytest.mark.parametrize("query", ["limit=1000000", "limit=0", "offset=-5"])
def test_list_reports_rejects_out_of_range_paging(client, query):
    """Unbounded, `?limit=1000000` dumped every job ever run in one request."""
    assert client.get(f"/reports?{query}").status_code == 422


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
def test_unknown_provider_is_404_not_500(client):
    resp = client.get("/providers/notreal/models", headers={"X-Provider-Key": "x"})
    assert resp.status_code == 404


def test_provider_models_requires_a_key_header(client):
    assert client.get("/providers/anthropic/models").status_code == 422


# ---------------------------------------------------------------------------
# Report lifecycle
# ---------------------------------------------------------------------------
def test_get_nonexistent_report(client):
    assert client.get("/report/nonexistent-job-id").status_code == 404


def test_generate_validates_the_query_length(client):
    resp = client.post("/report/generate", json={"query": "too short"})
    assert resp.status_code == 422


def _submit(client, **overrides):
    payload = {
        "query": "What are the key trends in artificial intelligence research in 2026?",
        "report_mode": "general",
        "max_rounds": 1,
        "token_budget": 20000,
        "api_key": "sk-ant-not-a-real-key",
        **overrides,
    }
    return client.post("/report/generate", json=payload)


def test_submitted_job_is_queued_and_pollable(client):
    resp = _submit(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"

    status = client.get(f"/report/status/{body['job_id']}")
    assert status.status_code == 200
    assert status.json()["query"].startswith("What are the key trends")


def test_failed_job_does_not_leak_the_exception_to_the_client(client, monkeypatch):
    """
    The stored error is a raw exception string. It is fine in the log and on the
    status feed, but /report/{id} used to interpolate it into the 500 detail —
    and the frontend renders `detail` verbatim.
    """
    import api.database as database

    resp = _submit(client)
    job_id = resp.json()["job_id"]

    database._memory_jobs[job_id]["status"] = "failed"
    database._memory_jobs[job_id]["error"] = (
        "postgresql://user:hunter2@10.0.0.4:5432/research_db is unreachable"
    )

    report = client.get(f"/report/{job_id}")
    assert report.status_code == 409
    assert "hunter2" not in report.text
    assert "10.0.0.4" not in report.text


def test_report_still_running_returns_202(client):
    import api.database as database

    job_id = _submit(client).json()["job_id"]
    database._memory_jobs[job_id]["status"] = "running"
    assert client.get(f"/report/{job_id}").status_code == 202


def test_completed_job_renders_markdown(client):
    """The API renders the report server-side so the UI has one string to show."""
    import api.database as database

    job_id = _submit(client).json()["job_id"]
    job = database._memory_jobs[job_id]
    job["status"] = "completed"
    job["report"] = {
        "title": "Test Report",
        "executive_summary": "A summary.",
        "key_findings": ["First finding", "Second finding"],
        "detailed_sections": {"Background": "Some background."},
        "citations": [
            {
                "index": 1,
                "url": "https://example.com/a",
                "title": "Example",
                "domain": "example.com",
                "accessed_date": "2026-08-20",
            }
        ],
        "contradictions": [],
        "confidence_assessment": "Moderate.",
        "limitations": "Small sample.",
        "quality": {
            "coverage_score": 0.8,
            "source_diversity_score": 0.7,
            "contradiction_rate": 0.1,
            "overall_quality_score": 0.75,
            "confidence_distribution": {
                "high": 1,
                "medium": 0,
                "low": 0,
                "contested": 0,
                "inconclusive": 0,
            },
            "re_research_rounds": 1,
            "total_sources_consulted": 1,
            "total_claims_extracted": 2,
            "claims_flagged_by_critic": 0,
            "claims_verified_by_fact_checker": 0,
            "converged": True,
            "convergence_note": "Report converged after 1 research round(s).",
        },
    }
    from datetime import UTC, datetime

    job["completed_at"] = datetime.now(UTC)
    job["duration_seconds"] = 12.5

    body = client.get(f"/report/{job_id}").json()
    md = body["report_markdown"]
    assert "# Test Report" in md
    assert "First finding" in md
    assert "example.com" in md
    assert body["quality"]["converged"] is True


def test_cors_allowlist_is_not_a_wildcard(client):
    """BYOK keys travel from the browser, so the origin allowlist matters."""
    resp = client.get("/health", headers={"Origin": "https://not-allowed.example"})
    assert resp.headers.get("access-control-allow-origin") != "*"
