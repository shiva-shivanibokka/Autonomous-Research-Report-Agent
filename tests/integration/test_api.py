"""Integration smoke tests for the FastAPI backend."""

import os
import pytest
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def test_health_check():
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics_endpoint():
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/metrics")
    assert resp.status_code == 200
    assert b"research_report_requests_total" in resp.content


def test_list_reports_empty():
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert "reports" in data


def test_get_nonexistent_report():
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/report/nonexistent-job-id")
    assert resp.status_code == 404


def test_submit_report_job():
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{API_URL}/report/generate",
            json={
                "query": "What are the key trends in artificial intelligence research in 2025?",
                "report_mode": "general",
                "max_rounds": 1,
                "token_budget": 20000,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"
