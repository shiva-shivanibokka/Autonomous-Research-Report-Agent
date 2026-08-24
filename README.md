# Autonomous Research Report Agent

[![CI](https://github.com/shiva-shivanibokka/Autonomous-Research-Report-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/shiva-shivanibokka/Autonomous-Research-Report-Agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

A multi-agent pipeline that researches the open web and writes **cited, quality-scored research reports** — with a self-improving critic loop, source triangulation, contradiction detection, and per-report cost accounting.

---

## ▶ [See it run →](https://autonomous-research-report-agent-shiv-a.vercel.app)

> ### This is a **replay**, not a live service.
>
> The page above plays back **one real run**, recorded on 24 August 2026 and
> committed to this repo as [`frontend/public/demo/run.json`](./frontend/public/demo/run.json).
> **There is no backend behind it.** Nothing you do on that page starts a
> research job, because there is no server to start one on.
>
> **Why not host it?** A report takes **four to five minutes**, launches headless
> Chromium, and holds the scraped pages in memory. No free tier will carry that,
> and a cold start that long reads as "broken" to anyone clicking a portfolio
> link. The honest options were a dead URL, a fake mockup, or running the real
> thing once and showing exactly what came out. This is the third.
>
> **What is real in it:** every agent message, token count, dollar figure,
> citation, quality score and the report itself — all produced by the actual
> pipeline against the live web and a live model. The only thing compressed is
> time: playback runs in about 30 seconds instead of 4m 34s, keeping the relative
> pacing of each stage. The page says all of this on its own banner, too.
>
> **Want it live?** It runs locally in two commands with no database —
> see [Run it locally](#run-it-locally). Bring your own LLM key.

---

## Recruiter TL;DR

- **What it is:** you ask a research question; seven specialized LLM agents decompose it, search and scrape the live web, triangulate sources, run a self-improving critic loop, fact-check flagged claims, and synthesize a structured report where every claim traces to a source and carries a confidence level.
- **Hardest problems solved:** orchestrating a stateful multi-agent graph with a conditional re-research loop under a hard token budget; a **provider-agnostic BYOK LLM layer** (Anthropic, OpenAI, Google, Groq) behind one interface; and making a pipeline that cannot be cheaply hosted still demonstrable and honest about it.
- **Stack:** Python · LangGraph · FastAPI · Pydantic · Celery/Redis (optional) · Postgres (optional) · Playwright · Prometheus/Grafana/OpenTelemetry · **Next.js 14 + TypeScript + Tailwind**.
- **Worth reading:** [Notable fixes](#notable-fixes) — nine defects found by *running* the thing, every one invisible to a green test suite.

---

## What the recorded run shows

It answers **"Do AI coding assistants actually make software developers more
productive?"** — chosen deliberately, because the evidence genuinely conflicts.
A question every source agrees on would demonstrate nothing this pipeline does
that a single search could not.

| | |
|---|---|
| Duration | 4m 34s (replayed in ~30s) |
| Rounds | **2 of 2** — the Critic judged round one insufficient and sent it back |
| Sources retrieved | 81, across 25 domains |
| Claims extracted | 60 |
| Citations in report | 30 |
| Tokens / cost | 78,591 · **$0.49** (Sonnet 4.5) |
| Overall quality | 58% · coverage 75% · source diversity 52% |
| Confidence spread | 6 high · 13 medium · **41 low** |
| Converged | **No** — the Critic's bar was still unmet when the rounds ran out |

**Those last three rows are the point.** The run surfaced the METR randomized
controlled trial, in which developers took **19% longer** with AI assistance
while *perceiving* a 20% speedup — against vendor studies claiming 20–50% gains.
It then scored its own output accordingly: mostly single-source claims, middling
diversity, did not converge. A tool that reported that as a clean success would
be lying about its own work.

Re-record it (or run a different question) with:

```bash
python scripts/record_demo_run.py --query "your question"
```

The recorder **refuses to write** if a key-shaped string appears anywhere in the
payload, if the artifact exceeds its size cap, or if the run produced no claims
or citations — an unhealthy run must not silently become the demo.

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
                         Next.js UI  ── BYOK provider / model / key ──┐
                              │  poll status / fetch report           │
                              ▼                                       │
                        FastAPI backend  ◀── GET /providers/{p}/models
                              │
           ┌──────────────────┴───────────────────┐
           │  JOB_BACKEND=inline (default)        │  JOB_BACKEND=celery
           │  asyncio task, in-process            │  Celery worker + Redis
           └──────────────────┬───────────────────┘
                              ▼
                       LangGraph pipeline
                              │
   Orchestrator ─▶ Search ─▶ Scrape ─▶ Analyst ─▶ Critic ──[needs more?]──▶ (loop)
   (decompose,     (Tavily)  (httpx +  (triangulate,   │ converged
    budget)                   Playwright) claims)      ▼
                                            Fact-checkers ─▶ Writer ─▶ Report
                                            (per flagged     (structured,
                                             claim)           cited, scored)
                              │
              Job store: Postgres, or in-process when DATABASE_URL is unset
              Prometheus · Grafana · OpenTelemetry/Jaeger (Compose only)
```

**The agents** (`agents/`):

| Agent | Does |
|---|---|
| **Orchestrator** | Decomposes the query into 3–5 non-overlapping sub-questions and allocates token budget; on re-research rounds it folds in the Critic's feedback. |
| **Search** | Parallel Tavily search per sub-question (`asyncio.gather`). |
| **Scraper** | Fetches and cleans pages with `httpx` + BeautifulSoup, falling back to headless **Playwright** for JS-heavy pages. Refuses private/loopback addresses and caps download size. |
| **Analyst** | Extracts claims and triangulates them across sources, tracking supporting vs contradicting counts. |
| **Critic** | Scores coverage/diversity/contradiction, flags weak claims, and decides whether another research round is warranted. |
| **Fact-checkers** | Spawned per flagged claim, each running its own targeted search. |
| **Writer** | Synthesizes a mode-specific structured report (general / competitive intelligence / investment thesis / literature review) with citations and a contradiction map. |

State flows through a single typed `ResearchState` (`agents/schemas.py`); every
agent boundary is a Pydantic model — no raw dicts between agents, and the final
report is validated against its mode's schema before it is served.

---

## Bring your own key (BYOK), multi-provider

The LLM layer (`agents/llm_client.py`) is provider-agnostic. Anthropic uses its
native SDK; **OpenAI, Groq and Google Gemini share one OpenAI-compatible path** —
three providers, one code branch. The key is chosen in the UI, sent per request,
and **never persisted**: it lives in memory only, never in the database. The
model dropdown is populated live from the provider's own `/models` endpoint using
that key.

Two pieces of provider drift are handled explicitly, because both are silent
failures otherwise:

- **Reasoning models** (o-series, gpt-5) reject `max_tokens` and require
  `max_completion_tokens`. A pattern catches the known ones; an actual 400
  teaches the cache for anything newer, so a model family invented after this was
  written costs one retry once rather than failing forever.
- **Sampling parameters** are not forwarded at all — several current models reject
  a non-default `temperature` with a hard 400. Determinism comes from the prompts.

---

## Repository layout

```
agents/            LangGraph pipeline: one file per agent + shared schemas
  llm_client.py    multi-provider BYOK client (contextvar-scoped creds)
  graph.py         graph topology + the self-improving loop
  report_format.py structured report -> Markdown
  tools/           Tavily search + the hardened scraper
api/               FastAPI app, job store, Celery worker, inline runner, metrics
config/            structured logging + OpenTelemetry setup
frontend/          Next.js 14 + TS + Tailwind UI
  public/demo/     the committed recording the hosted page replays
scripts/           record_demo_run.py — captures a real run as the demo
monitoring/        Prometheus, Grafana, Alertmanager, alert rules
tests/             83 tests; no network, no database, no API keys required
Dockerfile         multi-stage: base -> api / worker
docker-compose.yml full local stack (API, worker, Redis, Postgres, observability)
```

---

## Run it locally

### Quickest path — no database, no Docker

`DATABASE_URL` is optional. Leave it unset and jobs live in an in-process store.
Under the default `JOB_BACKEND=inline` that is not an approximation of the
database: the pipeline runs as an asyncio task inside the same process that
serves the status polls, so it is the same single-writer store without the
network hop. Jobs do not survive a restart, and `/health` reports
`"job_store": "memory"` so the substitution is never silent.

```bash
pip install -r requirements.txt
playwright install chromium              # optional — JS-page fallback
export TAVILY_API_KEY=tvly-...           # free tier, no card: tavily.com
uvicorn api.main:app --reload            # → http://localhost:8000/docs
```

In a second terminal:

```bash
cd frontend && npm install && npm run dev   # → http://localhost:3000
```

Open the UI, pick a provider, paste **your own** LLM key, and run a query.

### Full stack (Docker Compose)

Adds the durable path and the observability stack — Celery worker, Redis,
Postgres, Prometheus, Grafana, Jaeger:

```bash
cp .env.example .env          # add TAVILY_API_KEY; ANTHROPIC_API_KEY optional (BYOK)
docker compose up --build
# API      → http://localhost:8000  (docs at /docs)
# Grafana  → http://localhost:3000
```

### Configuration

| Variable | Default | Notes |
|---|---|---|
| `TAVILY_API_KEY` | — | **Required.** Web search. Free tier, no card. |
| `TAVILY_SEARCH_DEPTH` | `basic` | `basic` costs 1 Tavily credit per search, `advanced` costs 2. A report issues 10–15 searches, so this roughly halves credit use. |
| `DATABASE_URL` | _(unset)_ | Unset → in-process job store. Set it for durable, multi-instance storage. |
| `JOB_BACKEND` | `inline` | `inline` runs jobs in-process; `celery` dispatches to the worker (requires `DATABASE_URL` + Redis). |
| `ANTHROPIC_API_KEY` | _(unset)_ | Optional server-side fallback. Only reachable when the flag below is on. |
| `ALLOW_SERVER_KEY_FALLBACK` | `false` | Off by default: a request without its own key is **refused** rather than billed to the server's key. `/health` reports the current value. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS allowlist. BYOK keys come from the browser, so this matters in production. |
| `MAX_INLINE_JOBS` | `4` | Ceiling on concurrent in-process pipelines. |
| `LOG_LEVEL` | `INFO` | |

---

## Testing

```bash
pytest tests/unit          # pure logic: cost math, provider routing, JSON extraction, schemas
pytest tests/integration   # the API via TestClient — in-process store, no database, no keys
pytest                     # both (83 tests)
```

Neither suite needs a network, a database or an API key, so CI runs the whole
thing on every push, alongside a pinned `ruff` and a full frontend build.
`tests/unit/test_regressions.py` pins the specific defects listed below — each
one lived on a path that only executes with a live model, a real database or a
second research round, which is exactly why the original suite stayed green while
they shipped.

---

## Deployment

**What is deployed:** only the frontend, on Vercel, serving the committed
recording as a static replay. It is git-connected, so a push redeploys it.
`NEXT_PUBLIC_API_URL` is deliberately **unset** there — the UI infers replay mode
from having no backend URL while not being served from localhost, so a
misconfigured deploy cannot end up silently polling an address that does not
exist.

**What is not:** the backend. See the note at the top for why.

If you want your own full deployment, the two-tier shape it is built for:

- **Backend → any host that allows multi-minute requests** (Cloud Run, Fly, a VPS).
  Runs as a single scale-to-zero service in `JOB_BACKEND=inline`; the image
  honours `$PORT`. `DATABASE_URL` is optional but recommended for anything real.
  Set `ALLOWED_ORIGINS` to your frontend's origin.
- **Frontend → Vercel.** The `frontend/` directory deploys as a Next.js app; set
  `NEXT_PUBLIC_API_URL` to the backend URL and the replay steps aside for the
  real thing.

Step-by-step commands are in [`DEPLOY.md`](./DEPLOY.md).

---

## Observability

The Compose stack ships Prometheus metrics (`/metrics`), Grafana dashboards,
OpenTelemetry traces (a span per LLM call with model, tokens and cost),
structured JSON logs with per-request IDs, and Alertmanager rules. Tracing
no-ops when `OTLP_ENDPOINT` is unset, so nothing queues against an unreachable
collector in environments without one.

---

## Notable fixes

Found by **running the pipeline end to end against a live model**. None were
visible to the test suite, which was green throughout — each lived on a path that
only executes with a real model, a real database, or a second research round.

| What was wrong | Why it mattered |
|---|---|
| The Analyst prompt asked for `supporting_sources` as both a list of URLs and a count. Models sent the list; `int(...)` raised inside a bare `except: continue`. | **Every claim was silently discarded.** The pipeline paid for the tokens and produced empty reports. The recorded run now extracts 60. |
| `setup_logging()` paired `stdlib.add_logger_name` with `PrintLoggerFactory`, whose logger has no `.name`. | The first log call after startup raised — and that call is inside the API's own lifespan, so **the service could not finish booting** in any environment. |
| Both job runners wrote `completed_at` as an ISO **string** to a `TIMESTAMPTZ` column; asyncpg rejects that outright. | **No job could ever be marked complete** — and the error handler failed identically, leaving jobs stuck on `running` forever. |
| The activity log was persisted only at the end of a run. | The "live" agent feed stayed **empty for the whole multi-minute job**, then filled in at once. It now streams per graph node. |
| `increment_round()` cleared `search_outputs`, the only source `_build_citations` read. | Round-one sources vanished from the citation list while the claims citing them stayed in the report. |
| `converged` was hardcoded `True`; the Analyst and Fact-Checker discarded their `cost`. | Two headline numbers — the quality metric and the per-report cost — were **wrong by construction**. |
| The Orchestrator was the only LLM-calling agent with no error handling, and it runs first. | A rejected API key surfaced as an opaque 401 with an empty activity feed. |
| The Writer's output cap was 4096 tokens — too small for a full report. | Since the response is one JSON object, truncation lost **the entire report**, and presented as "malformed JSON" rather than "out of room". The repair retry inherited the same cap, so it could never have worked. |
| Citations were emitted one per line, and Markdown folds consecutive lines into one paragraph. | All 30 sources rendered as a single unreadable run-on block. |
| Every Tavily search was hardcoded to `advanced` (2 credits) rather than `basic` (1). | Doubled the credit cost of every run against a free tier, for a depth nothing required. |

---

## Limitations & roadmap

- **Inline mode ties a job to the instance running it.** Fine for an interactive
  session; the Celery path is the durable option. A Cloud Tasks worker is the
  natural next step for scale-to-zero durability.
- **Cost figures are best-effort estimates.** BYOK model lists are open-ended, so
  an unrecognised model falls back to a rough default — a guide, not a bill.
- **Web-sourced research inherits the open web's biases and recency gaps.** The
  confidence scores and contradiction map exist to make that visible, not to
  eliminate it.
- **The Fact-Checker only runs on claims the Critic flags.** A confidently wrong
  claim that nothing contradicts will pass through unverified.

---

## License

[MIT](./LICENSE) © Shivani Bokka

---

Built by **Shivani Bokka** · [github.com/shiva-shivanibokka](https://github.com/shiva-shivanibokka)
