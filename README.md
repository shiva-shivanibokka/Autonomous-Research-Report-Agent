# Autonomous Research Report Agent

A multi-agent pipeline that autonomously researches the open web and writes **cited, quality-scored research reports** — with a self-improving critic loop, source triangulation, contradiction detection, and per-report cost accounting.

> **Recruiter TL;DR**
> - **What it is:** you ask a research question; seven specialized LLM agents decompose it, search and scrape the live web, triangulate sources, run a self-improving critic loop, fact-check flagged claims, and synthesize a structured report where every claim is traced to a source and scored for confidence.
> - **Hardest problems solved:** orchestrating a stateful multi-agent graph with a conditional re-research loop under a hard token budget; a **provider-agnostic BYOK LLM layer** (Anthropic, OpenAI, Google, Groq) behind one interface; and a two-tier deploy (scale-to-zero backend + static frontend) that keeps a heavy Python+Playwright stack demoable and cheap.
> - **Stack:** Python · LangGraph · FastAPI · Celery/Redis (or in-process) · Postgres · Playwright · Prometheus/Grafana/OpenTelemetry · **Next.js 14 + TypeScript + Tailwind** frontend on Vercel · backend on Google Cloud Run.

**Live demo: a recording of a real run.** _(link added on deploy)_

This one cannot be hosted the ordinary way: a report takes several minutes,
launches headless Chromium and holds the scraped pages in memory, which no free
tier carries — and a cold start that long reads as "broken" to anyone clicking a
portfolio link. So rather than publish a dead URL or fake the product, the
pipeline was **run once for real** and its output committed:
[`frontend/public/demo/run.json`](./frontend/public/demo/run.json).

Everything the demo shows — the agent feed, the token counts, the dollar
figures, the citations, the quality scores and the report itself — is what
actually happened on that run. Playback is compressed to about half a minute
while keeping the relative pacing of each stage, and the page says so.

To run it live against your own key, see [Run it locally](#run-it-locally) —
two commands, no database.

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
Dockerfile         multi-stage: base -> api / worker
docker-compose.yml full local stack (API, worker, Redis, Postgres, observability)
```

---

## Run it locally

### Quickest path — no database, no Docker

`DATABASE_URL` is optional. Leave it unset and jobs are held in an in-process
store, which is not an approximation of the database in the default
`JOB_BACKEND=inline` mode: the pipeline runs as an asyncio task inside the same
process that serves the status polls, so it is the same single-writer store
without the network hop. Jobs do not survive a restart, and `/health` reports
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

Open the UI, pick a provider, paste **your own** LLM key, and run a query. The
key is sent per request and never stored.

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
| `TAVILY_SEARCH_DEPTH` | `basic` | `basic` costs 1 Tavily credit per search, `advanced` costs 2. A report issues 10-15 searches, so this roughly halves credit use. |
| `DATABASE_URL` | _(unset)_ | Unset → in-process job store. Set it for durable, multi-instance storage. |
| `JOB_BACKEND` | `inline` | `inline` runs jobs in-process; `celery` dispatches to the worker (requires `DATABASE_URL` + Redis). |
| `ANTHROPIC_API_KEY` | _(unset)_ | Optional server-side fallback. Only reachable when the flag below is on. |
| `ALLOW_SERVER_KEY_FALLBACK` | `false` | Off by default. A request without its own key is refused rather than billed to the server's key. `/health` reports the current value. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS allowlist. BYOK keys come from the browser, so this matters in production. |
| `MAX_INLINE_JOBS` | `4` | Ceiling on concurrent in-process pipelines. |
| `LOG_LEVEL` | `INFO` | |

---

## Testing

```bash
pytest tests/unit          # pure logic: cost math, provider routing, JSON extraction, schemas
pytest tests/integration   # the API via TestClient — in-process store, no database, no keys
pytest                     # both (77 tests)
```

Neither suite needs a network, a database, or an API key, so CI runs the whole
thing on every push. `tests/unit/test_regressions.py` pins the specific defects
listed under [Notable fixes](#notable-fixes) — each one lived on a path that only
executes with a live model or a second research round, which is exactly why the
original suite stayed green while they shipped.

---

## Deployment

There is no public deployment of this project (see the note at the top). If you
want your own, the two-tier shape it is built for:

- **Backend → any host that allows multi-minute requests** (Cloud Run, Fly, a VPS).
  Runs as a single scale-to-zero service in `JOB_BACKEND=inline` mode; the image
  honours `$PORT`. `DATABASE_URL` is optional but recommended for anything real.
  Set `ALLOWED_ORIGINS` to your frontend's origin.
- **Frontend → Vercel.** The `frontend/` directory deploys as a Next.js app; set
  `NEXT_PUBLIC_API_URL` to the backend URL.

Step-by-step commands are in [`DEPLOY.md`](./DEPLOY.md).

---

## Observability

The full Compose stack ships Prometheus metrics (`/metrics`), Grafana dashboards, OpenTelemetry traces to Jaeger (a span per LLM call with model, tokens, and cost), structured JSON logs with per-request IDs, and Alertmanager rules. On Cloud Run, tracing no-ops (no collector) and logs stream to Cloud Logging.

## The recorded run

The committed recording answers *"Do AI coding assistants actually make software
developers more productive?"* — chosen because the evidence genuinely conflicts,
which is the case the critic loop and contradiction map exist for. A question
every source agrees on would demo nothing this pipeline does that a single search
could not.

| | |
|---|---|
| Duration | 4m 34s |
| Rounds | 2 of 2 — the critic asked for a second pass |
| Sources retrieved | 81 |
| Claims extracted | 60 |
| Citations in report | 30 |
| Cost | $0.49 (Sonnet 4.5) |
| Converged | **No** — the critic's bar was still unmet when the rounds ran out |

That last row is reported rather than hidden. The run found the METR randomized
controlled trial, in which developers took **19% longer** with AI assistance
while *perceiving* a 20% speedup, alongside vendor studies claiming 20-50%
gains — and scored the result accordingly: 58% overall quality, 52% source
diversity, 41 of 60 claims at low confidence. A tool that reported this as a
clean success would be lying about its own output.

Re-record with:

```bash
python scripts/record_demo_run.py --query "your question"
```

It refuses to write if any key-shaped string appears in the payload, if the
artifact exceeds its size cap, or if the run produced no claims or citations —
an unhealthy run should not silently become the demo.

## Notable fixes

Found by running the pipeline end to end against a live model — none of them were
visible to the test suite, which was green the whole time.

| What was wrong | Why it mattered |
|---|---|
| The Analyst prompt asked for `supporting_sources` as both a list of URLs and a count. Models sent the list; `int(...)` raised inside a bare `except: continue`. | **Every claim was silently discarded.** The pipeline paid for the tokens and produced empty reports. Now 20+ claims per run. |
| `setup_logging()` paired `stdlib.add_logger_name` with `PrintLoggerFactory`, whose logger has no `.name`. | The first log call after startup raised, so **the API could not finish booting** in any environment. |
| Both job runners wrote `completed_at` as an ISO **string** to a `TIMESTAMPTZ` column; asyncpg rejects that outright. | **No job could ever be marked complete** — and the error handler failed the same way, leaving jobs on `running` forever. |
| The activity log was persisted only at the end of a run. | The "live" agent feed stayed **empty for the whole multi-minute job**. It now streams per graph node. |
| `increment_round()` cleared `search_outputs`, the only source `_build_citations` read. | Round-1 sources vanished from the citation list while the claims citing them stayed in the report. |
| `converged` was hardcoded `True`, and the Analyst and Fact-Checker discarded their `cost`. | Two headline numbers — the quality metric and the per-report cost — were **wrong by construction**. |
| The Orchestrator was the only LLM-calling agent with no error handling, and it runs first. | A rejected API key surfaced as an opaque 401 with an empty activity feed. |
| The Writer's output cap was 4096 tokens — too small for a full report, so a live run truncated mid-sentence. | Since the response is one JSON object, truncation lost **the entire report**, and presented as "malformed JSON" rather than "out of room". The repair retry inherited the same cap, so it could not have worked. |
| Every Tavily search was hardcoded to `advanced` (2 credits) rather than `basic` (1). | Doubled the credit cost of every run against a free tier, for a depth nothing required. |

## Limitations & roadmap

- Inline mode ties a job to the instance running it — fine for interactive demos (the polling UI keeps the instance warm); the Celery path is the durable option. A Cloud Tasks worker is the natural next step for scale-to-zero durability.
- Cost figures are **best-effort estimates** across providers (BYOK model lists are open-ended); treat them as a guide, not a bill.
- Web-sourced research inherits the open web's biases and recency gaps — the confidence scores and contradiction map exist to make that visible, not to eliminate it.

## License

[MIT](./LICENSE) © Shivani Bokka

---

Built by **Shivani Bokka** · [github.com/shiva-shivanibokka](https://github.com/shiva-shivanibokka)
