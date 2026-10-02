"""
Shared plumbing for the SOP evaluation: local models, caches, credit guard.

Nothing here changes pipeline logic. It swaps the transport underneath it:

* LLM calls go to a local Ollama server: the pipeline's own call_llm() (budget
  clamp, token/cost accounting, JSON parsing) is kept and only the HTTP call
  under it (`llm_client._call_openai_chat`) is replaced, with a fixed `seed`
  added. Responses are cached on disk keyed by (model, seed, max_tokens,
  messages). Identical prompts under the same seed therefore replay instead of
  re-generating — this is what lets the max_rounds=1 and max_rounds=2 runs
  share their (identical) round-1 orchestrator and analyst calls.
* Tavily searches are cached by (query, max_results, depth) and every cache
  miss is counted against a hard credit cap so the free tier can't be exceeded.
* Page fetches go through the pipeline's own scraper and are cached by URL.

The Tavily key is read in-process from the original repo's .env and never
printed.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

ORIGINAL_ENV = Path(
    r"<REPOS>\Autonomous-Research-Report-Agent\.env"
)
OLLAMA_URL = "http://localhost:11434"

GEN_MODEL = "qwen2.5:7b"  # ollama qwen2.5:7b (Q4_K_M); served with the server's auto num_ctx (32768 observed)
JUDGE_MODELS = ("llama3.1:8b", "gemma2:9b")  # Q4_K_M / Q4_0; judged with num_ctx 8192
JUDGE_NUM_CTX = 8192

# Hard ceiling on Tavily credits this evaluation may spend (free tier = 1000/mo).
TAVILY_CREDIT_CAP = int(os.environ.get("SOP_TAVILY_CAP", "950"))


def load_tavily_key() -> None:
    """Put TAVILY_API_KEY into os.environ from the original .env, silently."""
    if os.environ.get("TAVILY_API_KEY"):
        return
    for line in ORIGINAL_ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("TAVILY_API_KEY="):
            os.environ["TAVILY_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
            return
    raise RuntimeError("TAVILY_API_KEY not found in original .env")


# ---------------------------------------------------------------------------
# Tiny sqlite KV store
# ---------------------------------------------------------------------------
class KV:
    def __init__(self, name: str):
        self.path = CACHE / f"{name}.sqlite"
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()

    def get(self, k: str):
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (k, json.dumps(v))
        )
        self.db.commit()

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM kv").fetchone()[0]


def _h(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Per-run telemetry
# ---------------------------------------------------------------------------
class Telemetry:
    def __init__(self):
        self.llm_calls: list[dict] = []
        self.searches: list[dict] = []

    def summary(self) -> dict:
        return {
            "llm_calls": len(self.llm_calls),
            "llm_calls_from_cache": sum(c["cached"] for c in self.llm_calls),
            "input_tokens": sum(c["input_tokens"] for c in self.llm_calls),
            "output_tokens": sum(c["output_tokens"] for c in self.llm_calls),
            # Sum of the original generation times (also for cache replays):
            # the compute this run costs on this GPU if run from scratch.
            "llm_seconds": round(sum(c["seconds"] for c in self.llm_calls), 2),
            "tavily_searches": len(self.searches),
            "tavily_credits_if_uncached": sum(s["credits"] for s in self.searches),
            "tavily_credits_spent_now": sum(
                s["credits"] for s in self.searches if not s["cached"]
            ),
            "llm_calls_by_agent": _count_by(self.llm_calls, "agent"),
            "tokens_by_agent": _sum_by(self.llm_calls, "agent"),
        }


def _count_by(rows, key):
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def _sum_by(rows, key):
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + r["input_tokens"] + r["output_tokens"]
    return out


TELEMETRY: contextvars.ContextVar[Telemetry | None] = contextvars.ContextVar(
    "telemetry", default=None
)
CURRENT_AGENT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent", default="unknown"
)
SEED: contextvars.ContextVar[int] = contextvars.ContextVar("seed", default=0)


# ---------------------------------------------------------------------------
# LLM transport (Ollama native /api/chat, so per-request options such as
# num_ctx can be set without creating model aliases on the shared server)
# ---------------------------------------------------------------------------
_llm_cache = KV("llm")
# Ollama on this machine is shared with other workloads (OLLAMA_NUM_PARALLEL=1),
# so a request can queue for many minutes. Never time out; never auto-retry.
_http = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=None)
_gen_lock = asyncio.Lock()
RETRIES = [0]  # count of HTTP 5xx retries, reported in RESULTS


def _est_tokens(messages: list[dict]) -> int:
    return round(sum(len(m["content"]) for m in messages) / 3.6)


async def chat(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int,
    seed: int | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    agent: str | None = None,
) -> tuple[str, int, int, bool]:
    """One cached chat completion against Ollama. Returns (text, in, out, cached)."""
    seed = SEED.get() if seed is None else seed
    agent = agent or CURRENT_AGENT.get()
    key = _h([model, seed, temperature, max_tokens, messages])
    hit = _llm_cache.get(key)
    if hit is None:
        options = {"seed": seed, "num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        async with _gen_lock:
            t0 = time.perf_counter()
            # The shared server intermittently returns 500 while it reloads
            # runners for other workloads; retry those (and only those).
            for attempt in range(8):
                resp = await _http.post(
                    "/api/chat",
                    json={"model": model, "messages": messages, "stream": False,
                          "options": options, "keep_alive": "10m"},
                )
                if resp.status_code < 500:
                    break
                RETRIES[0] += 1
                await asyncio.sleep(20 + 20 * attempt)
            secs = time.perf_counter() - t0
        resp.raise_for_status()
        d = resp.json()
        hit = {
            "text": (d.get("message") or {}).get("content") or "",
            # prompt_eval_count can under-report when Ollama reuses its prompt
            # KV cache, so keep a character-based estimate alongside it.
            "prompt_eval_count": d.get("prompt_eval_count") or 0,
            "input_tokens_est": _est_tokens(messages),
            "output_tokens": d.get("eval_count") or 0,
            # Generation time as measured by Ollama (excludes queueing behind
            # other workloads on this shared GPU); wall time includes it.
            "seconds": (d.get("total_duration") or 0) / 1e9,
            "wall_seconds": secs,
            "finish_reason": d.get("done_reason"),
        }
        hit["input_tokens"] = max(hit["prompt_eval_count"], hit["input_tokens_est"])
        _llm_cache.put(key, hit)
        cached = False
    else:
        cached = True
    tel = TELEMETRY.get()
    if tel is not None:
        tel.llm_calls.append(
            {
                "agent": agent,
                "model": model,
                "input_tokens": hit["input_tokens"],
                "output_tokens": hit["output_tokens"],
                "seconds": hit["seconds"],
                "cached": cached,
                "finish_reason": hit.get("finish_reason"),
            }
        )
    return hit["text"], hit["input_tokens"], hit["output_tokens"], cached


# ---------------------------------------------------------------------------
# Tavily with cache + hard credit cap
# ---------------------------------------------------------------------------
_tavily_cache = KV("tavily")
_ledger = KV("tavily_ledger")


def credits_spent() -> int:
    return _ledger.get("spent") or 0


class CreditCapReached(RuntimeError):
    pass


def install(seed_default: int = 0):
    """Patch the pipeline's transports. Call once per process before running."""
    load_tavily_key()
    import logging

    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    import agents.fact_checker_agent as fact_checker_agent
    import agents.graph as graph
    import agents.llm_client as llm_client
    import agents.search_agent as search_agent
    import agents.tools.scraper_tool as scraper_tool
    import agents.tools.search_tool as search_tool
    import agents.writer_agent as writer_agent
    from agents.schemas import ScrapedPage, SearchResult

    # --- LLM: keep the pipeline's own call_llm (budget clamp, token/cost
    # accounting, JSON extraction) and replace only the HTTP call under it.
    async def _ollama_chat(client, model, max_tokens, messages):
        text, inp, out, _ = await chat(GEN_MODEL, messages, max_tokens=max_tokens)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=inp, completion_tokens=out),
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        )

    llm_client._call_openai_chat = _ollama_chat

    def _set_creds(provider, model, api_key=None):
        llm_client._creds_var.set(
            llm_client._LLMCreds(provider="ollama", model=GEN_MODEL, client=None)
        )

    graph.set_llm_creds = _set_creds

    orig_call_llm = llm_client.call_llm

    async def _tagged_call_llm(**kw):
        tok = CURRENT_AGENT.set(kw.get("agent_name", "unknown"))
        try:
            return await orig_call_llm(**kw)
        finally:
            CURRENT_AGENT.reset(tok)

    llm_client.call_llm = _tagged_call_llm
    writer_agent.call_llm = _tagged_call_llm

    # --- Tavily
    orig_search = search_tool.tavily_search

    async def cached_search(query, *, max_results=8, search_depth=None, **kw):
        depth = search_depth or search_tool.DEFAULT_SEARCH_DEPTH
        credits = 2 if depth == "advanced" else 1
        key = _h([query, max_results, depth, kw])
        hit = _tavily_cache.get(key)
        cached = hit is not None
        if not cached:
            if credits_spent() + credits > TAVILY_CREDIT_CAP:
                raise CreditCapReached(f"Tavily credit cap {TAVILY_CREDIT_CAP} reached")
            results = await orig_search(
                query, max_results=max_results, search_depth=depth, **kw
            )
            _ledger.put("spent", credits_spent() + credits)
            hit = [r.model_dump() for r in results]
            _tavily_cache.put(key, hit)
        tel = TELEMETRY.get()
        if tel is not None:
            tel.searches.append({"query": query, "credits": credits, "cached": cached})
        return [SearchResult(**r) for r in hit]

    search_tool.tavily_search = cached_search
    search_agent.tavily_search = cached_search
    fact_checker_agent.tavily_search = cached_search

    # --- Page fetches (the pipeline's scraper, cached by URL)
    orig_scrape = scraper_tool.scrape_page
    page_cache = KV("pages")

    async def cached_scrape(url, title=""):
        hit = page_cache.get(url)
        if hit is None:
            page = await orig_scrape(url, title)
            hit = page.model_dump()
            hit["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            page_cache.put(url, hit)
        hit = dict(hit)
        hit.pop("fetched_at", None)
        return ScrapedPage(**hit)

    scraper_tool.scrape_page = cached_scrape
    return SimpleNamespace(search=cached_search, scrape=cached_scrape, pages=page_cache)
