# Multi-stage Dockerfile for the Autonomous Research Report Agent.
# Stages: base -> worker, api. The `api` stage is LAST on purpose so
# `gcloud run deploy --source .` (which builds the final stage) deploys the API.
# docker-compose targets each stage by name, so ordering doesn't affect it.

# ---------------------------------------------------------------------------
# Base — shared dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for Playwright + asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium
RUN playwright install chromium --with-deps 2>/dev/null || true

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Worker stage (Celery — docker-compose only)
# ---------------------------------------------------------------------------
FROM base AS worker
CMD ["celery", "-A", "api.worker.celery_app", "worker", "--loglevel=info", "--concurrency=2"]

# ---------------------------------------------------------------------------
# API stage (final stage — what Cloud Run builds and runs)
# ---------------------------------------------------------------------------
FROM base AS api
EXPOSE 8000
# Honor Cloud Run's injected $PORT (defaults to 8000 locally). One worker keeps
# inline jobs and their asyncio tasks in a single process.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
