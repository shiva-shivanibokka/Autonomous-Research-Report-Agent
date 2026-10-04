"""
Shared plumbing for the SOP evaluation: model transports, caches, hard caps.

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

* Paid calls (Anthropic, pinned to claude-haiku-4-5-20251001) go through a
  persisted USD ledger with a pre-call worst-case check (eval_sop/budget.py).
  Each response is written to the disk cache as soon as it arrives, so a rerun
  after a crash replays it instead of paying twice. "Seeds" are replicate
  indices for Anthropic: the pipeline does not forward temperature
  (llm_client.call_llm) and the API has no seed, so different seeds are
  independent samples, not reproducible draws.

API keys are read in-process from an env file passed on the command line
(--env-file) and are never printed.
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

import anthropic
import httpx

from eval_sop import budget

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# SOP_OUT_DIR redirects caches/runs/results (used by smoke_test.py so a fake run
# never touches the real caches).
OUT_DIR = Path(os.environ["SOP_OUT_DIR"]) if os.environ.get("SOP_OUT_DIR") else HERE
CACHE = OUT_DIR / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"

GEN_MODELS = {
    # ollama qwen2.5:7b (Q4_K_M); served with the server's auto num_ctx (32768 observed)
    "ollama": "qwen2.5:7b",
    # Pinned snapshot. Every pipeline agent uses it: llm_client.call_llm takes
    # the model from the per-job creds, overriding the agents' REASON/FAST ids.
    "anthropic": "claude-haiku-4-5-20251001",
}
BACKEND = "ollama"
GEN_MODEL = GEN_MODELS[BACKEND]
JUDGE_MODELS = ("llama3.1:8b", "gemma2:9b")  # local Ollama; Q4_K_M / Q4_0; num_ctx 8192
JUDGE_NUM_CTX = 8192

# Hard ceiling on Tavily credits this evaluation may spend. The free tier is
# 1000/month and may be shared with other projects using the same key, so this
# stays well below it; check headroom first with eval_sop/tavily_usage.py.
TAVILY_CREDIT_CAP = int(os.environ.get("SOP_TAVILY_CAP", "600"))

_ANTHROPIC_KEY: str | None = None


def load_keys(env_file: Path | None) -> None:
    """Read TAVILY_API_KEY / ANTHROPIC_API_KEY from `env_file` in-process, silently."""
    global _ANTHROPIC_KEY
    if env_file is None:
        return
    from dotenv import dotenv_values

    vals = dotenv_values(env_file)
    if vals.get("TAVILY_API_KEY") and not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = vals["TAVILY_API_KEY"]
    if vals.get("ANTHROPIC_API_KEY"):
        _ANTHROPIC_KEY = vals["ANTHROPIC_API_KEY"]


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
            "usd_spent_now": round(
                sum(c.get("cost_usd", 0.0) for c in self.llm_calls if not c["cached"]), 6
            ),
            "usd_if_uncached": round(sum(c.get("cost_usd", 0.0) for c in self.llm_calls), 6),
            "models": sorted({c["model"] for c in self.llm_calls}),
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
RUN_KEY: contextvars.ContextVar[str] = contextvars.ContextVar("run_key", default="-")


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


class TransportError(Exception):
    """A provider call failed for good. Deliberately not one of the exception
    types llm_client.call_llm's tenacity layer retries, so the only retries
    are the bounded ones below."""


# Raw transport errors that can escape the SDK mid-stream. anthropic 1.x
# streams over httpx2, whose exceptions do not derive from httpx's.
try:
    import httpx2

    _TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (httpx.TransportError, httpx2.TransportError)
except ImportError:  # older SDKs use httpx only
    _TRANSPORT_ERRORS = (httpx.TransportError,)

MAX_RETRIES = 2  # per call; only 429 / 5xx / connection errors. Other 4xx fail fast.
_sleep = asyncio.sleep  # patched in tests
LEDGER: budget.UsdLedger | None = None
_anthropic_client = None


def set_anthropic_client(client) -> None:
    """Tests inject a fake client here."""
    global _anthropic_client
    _anthropic_client = client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not _ANTHROPIC_KEY:
            raise TransportError("no ANTHROPIC_API_KEY in --env-file")
        # max_retries=0: the SDK's own retries would be invisible to the ledger.
        _anthropic_client = anthropic.AsyncAnthropic(
            api_key=_ANTHROPIC_KEY, max_retries=0, timeout=600.0
        )
    return _anthropic_client


def _retry_after(exc) -> float | None:
    try:
        return min(60.0, float(exc.response.headers.get("retry-after")))
    except Exception:  # noqa: BLE001 - header absent or unparsable
        return None


async def _anthropic_generate(model, messages, max_tokens, temperature, agent):
    budget.price_of(model)  # refuses models outside the allow list
    if LEDGER is None:
        raise TransportError("USD ledger not initialised: install(backend='anthropic')")
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [m for m in messages if m["role"] != "system"],
    }
    if system:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    client = _get_anthropic_client()
    run = RUN_KEY.get()
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        # Cap check + pending worst-case row in one transaction, committed
        # before the request is sent (see budget.UsdLedger.reserve).
        worst, row = LEDGER.reserve(model=model, messages=messages, max_tokens=max_tokens,
                                    run=run, agent=agent)
        t0 = time.perf_counter()
        try:
            # Streaming: the SDK refuses non-streaming requests whose max_tokens
            # implies a >10-minute response (the Writer's retry asks for 24k).
            async with client.messages.stream(**kwargs) as stream:
                msg = await stream.get_final_message()
        except anthropic.RateLimitError as e:  # HTTP 429 before generation: not billed
            last_exc = e
            LEDGER.settle(row, status="429", cost_usd=0.0)
            wait = _retry_after(e) or 5.0 * (attempt + 1)
        except anthropic.APIStatusError as e:
            code = getattr(e, "status_code", None)
            etype = getattr(e, "type", None)
            if code in (401, 402, 403) or etype == "billing_error":
                LEDGER.settle(row, status=str(code), cost_usd=0.0, note=str(etype))
                budget.trip(f"Anthropic {code} {etype}: stopping all paid calls")
            if code == 200 or etype in ("overloaded_error", "api_error"):
                # An error event inside an accepted stream (HTTP 200), or an
                # overload / API error: retryable, and possibly billed.
                last_exc = e
                LEDGER.settle(row, status=f"stream_error:{etype}", cost_usd=worst,
                              note="charged at worst case: billing unknown")
                wait = _retry_after(e) or 5.0 * (attempt + 1)
            elif code is not None and code >= 500:  # HTTP-level rejection: not billed
                last_exc = e
                LEDGER.settle(row, status=str(code), cost_usd=0.0)
                wait = _retry_after(e) or 5.0 * (attempt + 1)
            else:  # any other 4xx: rejected, not billed, fail fast
                LEDGER.settle(row, status=str(code), cost_usd=0.0, note=str(e)[:200])
                raise TransportError(f"Anthropic {code}: {str(e)[:300]}") from e
        except (anthropic.APIConnectionError, *_TRANSPORT_ERRORS) as e:
            # Timeouts, dropped / reset connections, protocol errors — before or
            # during the stream. Billing unknown: keep the worst-case charge.
            last_exc = e
            LEDGER.settle(row, status="connection_error", cost_usd=worst,
                          note=f"charged at worst case: {type(e).__name__}")
            wait = 5.0 * (attempt + 1)
        else:
            u = msg.usage
            cw = getattr(u, "cache_creation_input_tokens", 0) or 0
            cr = getattr(u, "cache_read_input_tokens", 0) or 0
            cost = budget.call_cost(model, u.input_tokens, u.output_tokens, cw, cr)
            LEDGER.settle(
                row, status="ok", cost_usd=cost,
                input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                cache_write=cw, cache_read=cr, request_id=getattr(msg, "id", None),
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            return {
                "text": text,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cost_usd": cost,
                "seconds": time.perf_counter() - t0,
                "finish_reason": "length" if msg.stop_reason == "max_tokens" else msg.stop_reason,
                "model_reported": getattr(msg, "model", model),
            }
        # Anything else (KeyboardInterrupt, CancelledError, an unexpected
        # exception) propagates with the row still 'pending' at worst case.
        if attempt < MAX_RETRIES:
            await _sleep(wait)
    raise TransportError(f"gave up after {MAX_RETRIES} retries: {str(last_exc)[:300]}")


async def _ollama_generate(model, messages, max_tokens, seed, temperature, num_ctx):
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
        "cost_usd": 0.0,
    }
    hit["input_tokens"] = max(hit["prompt_eval_count"], hit["input_tokens_est"])
    return hit


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
    """One cached chat completion. Returns (text, in, out, cached)."""
    budget.check_stop()
    seed = SEED.get() if seed is None else seed
    agent = agent or CURRENT_AGENT.get()
    key = _h([model, seed, temperature, max_tokens, messages])
    hit = _llm_cache.get(key)
    if hit is None:
        if model.startswith("claude"):
            hit = await _anthropic_generate(model, messages, max_tokens, temperature, agent)
        else:
            hit = await _ollama_generate(model, messages, max_tokens, seed, temperature, num_ctx)
        _llm_cache.put(key, hit)  # persisted at once: a later crash can't re-bill this call
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
                "cost_usd": hit.get("cost_usd", 0.0),
            }
        )
    return hit["text"], hit["input_tokens"], hit["output_tokens"], cached


# ---------------------------------------------------------------------------
# Tavily with cache + hard credit cap
# ---------------------------------------------------------------------------
_tavily_cache = KV("tavily")


def credits_spent() -> int:
    return _credit_ledger().spent()


class CreditCapReached(budget.BudgetStop):  # noqa: N818
    """Tavily credit cap. A BudgetStop, so the agents' `except Exception` can't swallow it."""


_ORIGINALS: dict = {}
STATE_DIR = budget.default_state_dir()  # tests monkeypatch this
_LOCKS: dict[Path, budget.ProcessLock] = {}
_CREDITS: dict[Path, budget.CreditLedger] = {}


def ledger_path() -> Path:
    """The project's one USD ledger, outside any checkout (see budget.default_state_dir)."""
    return STATE_DIR / "usd_ledger.sqlite"


def tavily_ledger_path() -> Path:
    return STATE_DIR / "tavily_credits.sqlite"


def lock_path() -> Path:
    return STATE_DIR / "run.lock"


def _credit_ledger() -> budget.CreditLedger:
    path = tavily_ledger_path()
    if path not in _CREDITS:
        _CREDITS[path] = budget.CreditLedger(path)
    return _CREDITS[path]


def install(
    backend: str = "ollama", env_file: Path | None = None, usd_cap: float | None = None
):
    """Patch the pipeline's transports. Call once per process before running."""
    global BACKEND, GEN_MODEL, LEDGER
    if backend not in GEN_MODELS:
        raise ValueError(f"backend must be one of {sorted(GEN_MODELS)}")
    BACKEND, GEN_MODEL = backend, GEN_MODELS[backend]
    # One spending process per project at a time (USD and Tavily ledgers are
    # shared by every checkout); held until this process exits.
    lp = lock_path()
    if lp not in _LOCKS:
        _LOCKS[lp] = budget.ProcessLock(lp)
    _LOCKS[lp].acquire()
    load_keys(env_file)
    if backend == "anthropic" and LEDGER is None:
        cap = usd_cap if usd_cap is not None else budget.cap_from_env()
        LEDGER = budget.UsdLedger(ledger_path(), cap)
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
            # Not "anthropic": call_llm must take its OpenAI-compatible branch,
            # whose HTTP call (_call_openai_chat) is the one replaced above. The
            # real provider is chosen in chat() from the model id.
            llm_client._LLMCreds(provider=f"eval-{BACKEND}", model=GEN_MODEL, client=None)
        )

    graph.set_llm_creds = _set_creds

    # install() may run more than once per process (tests, canary + run); wrap
    # the ORIGINAL functions every time, never a previous wrapper.
    orig = _ORIGINALS.setdefault("call_llm", llm_client.call_llm)
    orig_call_llm = orig

    async def _tagged_call_llm(**kw):
        tok = CURRENT_AGENT.set(kw.get("agent_name", "unknown"))
        try:
            return await orig_call_llm(**kw)
        finally:
            CURRENT_AGENT.reset(tok)

    llm_client.call_llm = _tagged_call_llm
    writer_agent.call_llm = _tagged_call_llm

    # --- Tavily
    orig_search = _ORIGINALS.setdefault("tavily_search", search_tool.tavily_search)

    async def cached_search(query, *, max_results=8, search_depth=None, **kw):
        depth = search_depth or search_tool.DEFAULT_SEARCH_DEPTH
        credits = 2 if depth == "advanced" else 1
        key = _h([query, max_results, depth, kw])
        hit = _tavily_cache.get(key)
        cached = hit is not None
        if not cached:
            budget.check_stop()
            # Book the credits atomically (one BEGIN IMMEDIATE transaction, so
            # parallel searches and other processes see each other) and BEFORE
            # awaiting; a failed search still counts (conservative).
            if not _credit_ledger().reserve(credits, TAVILY_CREDIT_CAP):
                msg = f"Tavily credit cap {TAVILY_CREDIT_CAP} reached"
                budget.STOP["reason"] = msg
                raise CreditCapReached(msg)
            results = await orig_search(
                query, max_results=max_results, search_depth=depth, **kw
            )
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
    orig_scrape = _ORIGINALS.setdefault("scrape_page", scraper_tool.scrape_page)
    page_cache = KV("pages")

    async def cached_scrape(url, title=""):
        hit = page_cache.get(url)
        # A failed fetch is cached only for the current process (so c and d,
        # which run in one process, see identical inputs) and marked; any
        # later invocation retries it instead of inheriting the failure.
        stale_failure = (
            hit is not None
            and (hit.get("scrape_error") or not hit.get("content"))
            and hit.get("failed_in_pid") != os.getpid()
        )
        if hit is None or stale_failure:
            page = await orig_scrape(url, title)
            hit = page.model_dump()
            hit["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            failed = bool(page.scrape_error) or not page.content
            hit["failed_in_pid"] = os.getpid() if failed else None
            page_cache.put(url, hit)
        hit = dict(hit)
        hit.pop("fetched_at", None)
        hit.pop("failed_in_pid", None)
        return ScrapedPage(**hit)

    scraper_tool.scrape_page = cached_scrape
    return SimpleNamespace(search=cached_search, scrape=cached_scrape, pages=page_cache)
