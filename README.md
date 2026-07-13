# Autonomous Research Report Agent

A multi-agent pipeline that autonomously researches the open web and writes **cited, quality-scored research reports** — with a self-improving critic loop, source triangulation, contradiction detection, and per-report cost accounting.

> **Recruiter TL;DR**
> - **What it is:** you ask a research question; seven specialized LLM agents decompose it, search and scrape the live web, triangulate sources, run a self-improving critic loop, fact-check flagged claims, and synthesize a structured report where every claim is traced to a source and scored for confidence.
> - **Hardest problems solved:** orchestrating a stateful multi-agent graph with a conditional re-research loop under a hard token budget; a **provider-agnostic BYOK LLM layer** (Anthropic, OpenAI, Google, Groq) behind one interface; and a two-tier deploy (scale-to-zero backend + static frontend) that keeps a heavy Python+Playwright stack demoable and cheap.
> - **Stack:** Python · LangGraph · FastAPI · Celery/Redis (or in-process) · Postgres · Playwright · Prometheus/Grafana/OpenTelemetry · **Next.js 14 + TypeScript + Tailwind** frontend on Vercel · backend on Google Cloud Run.

**Live demo:** _deploying — see [Deployment](#deployment)._

---

## Why not just ask a chatbot?

| | General chatbot | Research Agent |
|---|---|---|
| Sources | whatever's in weights / one search | **decomposed sub-questions**, parallel web search + scrape |
| Trust | "trust me" | every claim carries **supporting/contradicting source counts** and a confidence level |
| Disagreement | smoothed over | **contradiction map** — surfaces where sources conflict and how it was resolved |
| Quality | unknown | **coverage, source-diversity, contradiction-rate, overall** scores per report |
| Rigor | single pass | **self-improving critic loop** re-researches weak claims until it converges or hits budget |
| Cost | opaque | **per-agent token + USD accounting** on every run |
| Model | fixed | **bring your own key** — Anthropic, OpenAI, Google, or Groq, chosen in the UI |

---

## Architecture

```
                         Next.js UI (Vercel)  ── BYOK provider/model/key ──┐
                                 │  poll status / fetch report             │
                                 ▼                                         │
                        FastAPI backend (Cloud Run)  ◀── GET /providers/{p}/models
                                 │
              ┌──────────────────┴───────────────────┐
              │  JOB_BACKEND=inline (Cloud Run)       │  JOB_BACKEND=celery (Compose)
              │  asyncio task in-process              │  Celery worker + Redis queue
              └──────────────────┬───────────────────┘
                                 ▼
                         LangGraph pipeline
                                 │
   Orchestrator ─▶ Search ─▶ Scrape ─▶ Analyst ─▶ Critic ──[needs more?]──▶ (loop)
   (decompose,     (Tavily)  (httpx +  (triangulate,   │ converged
    budget)                   Playwright) claims)       ▼
                                              Fact-checkers ─▶ Writer ─▶ Report
                                              (per flagged    (structured,
                                               claim)          cited, scored)
                                 │
                    Postgres (jobs, activity log, reports)
                    Prometheus · Grafana · OpenTelemetry/Jaeger
```

**The agents** (`agents/`):
- **Orchestrator** — decomposes the query into 3–5 non-overlapping sub-questions and allocates token budget; on re-research rounds it folds in the Critic's feedback.
- **Search** — parallel Tavily search per sub-question (`asyncio.gather`).
- **Scraper** — fetches and cleans pages with `httpx` + BeautifulSoup, falling back to headless **Playwright** for JS-heavy pages.
- **Analyst** — extracts claims and triangulates them across sources, tracking supporting vs contradicting counts.
- **Critic** — scores coverage/diversity/contradiction, flags weak claims, and decides whether another research round is worth it.
- **Fact-checkers** — spawned per flagged claim to verify against sources.
- **Writer** — synthesizes a mode-specific structured report (general / competitive intelligence / investment thesis / literature review) with citations and a contradiction map.

State flows through a single typed `ResearchState` (`agents/schemas.py`); every agent boundary is a Pydantic model — no raw dicts between agents.

---

## Bring your own key (BYOK), multi-provider

The LLM layer (`agents/llm_client.py`) is provider-agnostic. Anthropic uses its native SDK; **OpenAI, Groq, and Google Gemini share one OpenAI-compatible path** (three providers, one code branch). The user's key is chosen in the UI, sent per request, and **never persisted** — it lives only in memory (inline mode) or transiently on the queue (Celery mode), never in the database. The model dropdown is populated live from the provider's own `/models` endpoint using that key.

---

## Repository layout

```
agents/            LangGraph pipeline: one file per agent + shared schemas
  llm_client.py    multi-provider BYOK client (contextvar-scoped creds)
  graph.py         graph topology + the self-improving loop
  report_format.py structured report -> Markdown
api/               FastAPI app, DB layer, Celery worker, inline runner, metrics
config/            structured logging + OpenTelemetry setup
frontend/          Next.js 14 + TS + Tailwind UI (Vercel)
monitoring/        Prometheus, Grafana, Alertmanager, alert rules
tests/             unit + integration tests
Dockerfile         multi-stage: base -> api / worker / gradio
docker-compose.yml full local stack (API, worker, Redis, Postgres, observability)
```

---

## Run it locally

**Full stack (Docker Compose)** — API + Celery worker + Redis + Postgres + Prometheus/Grafana/Jaeger:

```bash
cp .env.example .env          # add TAVILY_API_KEY; ANTHROPIC_API_KEY optional (BYOK)
docker compose up --build
# API      → http://localhost:8000  (docs at /docs)
# Grafana  → http://localhost:3000
```

**Frontend** (points at the local API by default):

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev                   # → http://localhost:3000
```

**Backend only, no Redis/Celery** (inline mode — needs just Postgres):

```bash
JOB_BACKEND=inline DATABASE_URL=postgresql://... TAVILY_API_KEY=... \
  uvicorn api.main:app --reload
```

---

## Testing

```bash
pytest tests/unit -v          # pure, no network (cost math, provider routing, formatting, schemas)
pytest tests/integration      # smoke tests against a running API
```

---

## Deployment

Two-tier, chosen so a heavy Python + Playwright stack stays cheap and demoable:

- **Backend → Google Cloud Run.** Runs as a single scale-to-zero service in `JOB_BACKEND=inline` mode (no Redis/Celery); the image honors Cloud Run's `$PORT`. Needs a managed Postgres (`DATABASE_URL`, e.g. Neon) and `TAVILY_API_KEY`.
- **Frontend → Vercel.** The `frontend/` directory deploys as a Next.js app; set `NEXT_PUBLIC_API_URL` to the Cloud Run URL.

Step-by-step commands are in [`DEPLOY.md`](./DEPLOY.md).

---

## Observability

The full Compose stack ships Prometheus metrics (`/metrics`), Grafana dashboards, OpenTelemetry traces to Jaeger (a span per LLM call with model, tokens, and cost), structured JSON logs with per-request IDs, and Alertmanager rules. On Cloud Run, tracing no-ops (no collector) and logs stream to Cloud Logging.

## Limitations & roadmap

- Inline mode ties a job to the instance running it — fine for interactive demos (the polling UI keeps the instance warm); the Celery path is the durable option. A Cloud Tasks worker is the natural next step for scale-to-zero durability.
- Cost figures are **best-effort estimates** across providers (BYOK model lists are open-ended); treat them as a guide, not a bill.
- Web-sourced research inherits the open web's biases and recency gaps — the confidence scores and contradiction map exist to make that visible, not to eliminate it.

## License

[MIT](./LICENSE) © Shivani Bokka
