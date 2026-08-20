"""
Centralized, multi-provider LLM client (BYOK — bring your own key).

Supports Anthropic (native SDK) plus OpenAI, Groq, and Google Gemini via their
OpenAI-compatible endpoints — so three of the four providers share one code path.

Per-job credentials (provider + model + the user's API key) are set once via
`set_llm_creds()` and carried on a contextvar, mirroring how this codebase
already scopes request data with structlog contextvars. Agents keep calling
`call_llm(...)` unchanged; the active provider/model/key come from the contextvar.
If no creds are set (tests, local dev), it falls back to Anthropic with the
`model` arg and the server's ANTHROPIC_API_KEY from the environment.

Also provides:
- Token counting and best-effort cost estimation per call
- Prompt-instructed JSON structured output (provider-agnostic)
- Retry with exponential backoff (tenacity), across provider error types
- OpenTelemetry span per call
"""

from __future__ import annotations

import contextvars
import json
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
import openai
import structlog
from opentelemetry import trace
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
SUPPORTED_PROVIDERS = ("anthropic", "openai", "google", "groq")

# OpenAI-compatible base URLs. `None` = the openai SDK default (api.openai.com).
_OPENAI_COMPATIBLE_BASE: dict[str, str | None] = {
    "openai": None,
    "groq": "https://api.groq.com/openai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

# Retry only on transient errors, across both SDKs' exception hierarchies.
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    openai.RateLimitError,
    openai.APIConnectionError,
)

# ---------------------------------------------------------------------------
# Best-effort cost table (per 1M tokens, USD), keyed by model id (exact or prefix).
# With BYOK + dynamic model lists this can't be exhaustive — it's an *estimate*.
# Unknown models fall back to a rough default; surface cost as "estimated" in UI.
# ---------------------------------------------------------------------------
_COST_TABLE: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Groq (Llama)
    "llama-3.3-70b": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b": {"input": 0.05, "output": 0.08},
    # Google Gemini
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}
_DEFAULT_PRICE = {"input": 1.00, "output": 3.00}

# Default model when a caller doesn't specify one (server-key fallback path).
FAST_MODEL = "claude-haiku-4-5"
REASON_MODEL = "claude-sonnet-4-5"

# ---------------------------------------------------------------------------
# OpenAI parameter drift
#
# OpenAI's reasoning models (o1/o3/o4-…, gpt-5-…) removed `max_tokens` and take
# `max_completion_tokens` instead — sending the old name is a hard 400, not a
# warning. The model list in the UI comes live from /v1/models, so these ids show
# up in the dropdown whether or not this code knows about them.
#
# Two layers, because a static pattern can only know about today's model names:
#   1. the regex skips the doomed first call for ids we recognise, and
#   2. `_MAX_TOKENS_PARAM` learns from an actual 400 for anything we don't,
#      so a model family invented after this was written costs one retry, once,
#      rather than failing every call forever.
# ---------------------------------------------------------------------------
_REASONING_MODEL_RE = re.compile(r"^(?:o\d|gpt-5)")

# model id -> the param name known to work. Populated by the regex up front and
# corrected by _call_openai_chat when the API disagrees.
_MAX_TOKENS_PARAM: dict[str, str] = {}


def _max_tokens_param_for(model: str) -> str:
    """Which max-tokens parameter name this model wants."""
    cached = _MAX_TOKENS_PARAM.get(model)
    if cached is not None:
        return cached
    return "max_completion_tokens" if _REASONING_MODEL_RE.match(model) else "max_tokens"


async def _call_openai_chat(client: Any, model: str, max_tokens: int, messages: list):
    """
    One OpenAI-compatible chat call, retrying once if the server rejects the
    max-tokens parameter name. The correction is cached per model id.
    """
    param = _max_tokens_param_for(model)
    try:
        resp = await client.chat.completions.create(
            model=model, messages=messages, **{param: max_tokens}
        )
    except openai.BadRequestError as exc:
        other = "max_completion_tokens" if param == "max_tokens" else "max_tokens"
        # Only retry when the complaint is actually about this parameter —
        # otherwise a genuine 400 (bad key, bad model) would be retried blind.
        if other not in str(exc):
            raise
        log.info("openai_max_tokens_param_switched", model=model, to=other)
        resp = await client.chat.completions.create(
            model=model, messages=messages, **{other: max_tokens}
        )
        param = other
    _MAX_TOKENS_PARAM[model] = param
    return resp


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
def extract_json(text: str, expect: type | None = None) -> Any:
    """
    Pull the first complete JSON value out of an LLM reply.

    Models add prose, wrap output in ```json fences, or append a note after the
    closing brace no matter how firmly the prompt says not to. `raw_decode`
    parses from the first `{`/`[` and stops at the end of that value, so trailing
    commentary is ignored rather than fed to the parser.

    `expect` restricts which candidates count. Without it, a reply whose
    top-level object is malformed can still "succeed" by returning some nested
    array or object found further along — a plausible-looking value that is not
    the thing the caller asked for. Passing `expect=dict` turns that near-miss
    back into the ValueError it should be.

    Raises ValueError if there is no such value — callers decide what a missing
    report is worth, which is not something a parser should guess.
    """
    if not text:
        raise ValueError("empty response")

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if expect is not None and not isinstance(value, expect):
            continue
        return value
    raise ValueError(
        f"no JSON {expect.__name__ if expect else 'value'} found in response"
    )


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Best-effort USD cost for a call. Exact match, then longest prefix, then default.

    Longest-first matters: dated model ids like `gpt-4o-mini-2024-07-18` prefix-match
    both `gpt-4o-mini` and, if the table were ordered differently, `gpt-4o` — which
    is 17x the price. Iterating in insertion order happened to give the right answer
    only because the longer keys were written first, which is not a property anyone
    would think to preserve when adding a model.
    """
    prices = _COST_TABLE.get(model)
    if prices is None:
        prices = next(
            (
                p
                for key, p in sorted(
                    _COST_TABLE.items(), key=lambda kv: len(kv[0]), reverse=True
                )
                if model.startswith(key)
            ),
            _DEFAULT_PRICE,
        )
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]


# ---------------------------------------------------------------------------
# Per-job credentials (contextvar-scoped)
# ---------------------------------------------------------------------------
@dataclass
class _LLMCreds:
    provider: str
    model: str
    client: Any  # a constructed AsyncAnthropic or AsyncOpenAI, keyed to the user's key


_creds_var: contextvars.ContextVar[_LLMCreds | None] = contextvars.ContextVar(
    "llm_creds", default=None
)


def _build_client(provider: str, api_key: str | None) -> Any:
    provider = provider.lower()
    if provider == "anthropic":
        # api_key=None → SDK reads ANTHROPIC_API_KEY from the environment.
        return (
            anthropic.AsyncAnthropic(api_key=api_key)
            if api_key
            else anthropic.AsyncAnthropic()
        )
    if provider in _OPENAI_COMPATIBLE_BASE:
        return openai.AsyncOpenAI(
            api_key=api_key, base_url=_OPENAI_COMPATIBLE_BASE[provider]
        )
    raise ValueError(
        f"Unsupported provider: {provider!r}. One of {SUPPORTED_PROVIDERS}."
    )


def set_llm_creds(provider: str, model: str, api_key: str | None = None) -> None:
    """
    Set the BYOK provider/model/key for the current job (one client per job).
    Call once at pipeline start; every subsequent call_llm() in this context uses it.
    """
    provider = (provider or "anthropic").lower()
    _creds_var.set(
        _LLMCreds(
            provider=provider, model=model, client=_build_client(provider, api_key)
        )
    )


async def list_models(provider: str, api_key: str) -> list[str]:
    """List available model ids for a provider using the caller's key. Also validates the key."""
    provider = provider.lower()
    client = _build_client(provider, api_key)
    ids: list[str] = []
    async for m in client.models.list():
        mid = m.id
        # Gemini's OpenAI-compatible /models returns "models/gemini-..." — strip for chat calls.
        ids.append(mid.split("/", 1)[1] if mid.startswith("models/") else mid)
    return sorted(ids)


# ---------------------------------------------------------------------------
# Raw text completion
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def call_llm(
    *,
    system: str,
    user: str,
    model: str = REASON_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.2,  # ponytail: accepted for back-compat but NOT forwarded —
    # some BYOK models (Anthropic Opus 4.8 / Sonnet 5, OpenAI reasoning models) reject
    # non-default sampling params with a 400. Prompt instructions carry the determinism.
    agent_name: str = "unknown",
    token_budget_remaining: int | None = None,
) -> tuple[str, int, int, float]:
    """
    Call the active provider's LLM and return (text, input_tokens, output_tokens, cost_usd).

    Provider/model/key come from the per-job creds set via set_llm_creds(); if unset,
    falls back to Anthropic + the `model` arg + the server's ANTHROPIC_API_KEY.
    max_tokens is capped to token_budget_remaining when provided.
    """
    if token_budget_remaining is not None:
        max_tokens = min(max_tokens, max(256, token_budget_remaining))

    creds = _creds_var.get()
    if creds is not None:
        provider, model_id, client = creds.provider, creds.model, creds.client
    else:
        provider, model_id, client = "anthropic", model, anthropic.AsyncAnthropic()

    with tracer.start_as_current_span(f"llm.{agent_name}") as span:
        span.set_attribute("llm.provider", provider)
        span.set_attribute("llm.model", model_id)
        span.set_attribute("llm.agent", agent_name)
        span.set_attribute("llm.max_tokens", max_tokens)

        t0 = time.perf_counter()

        if provider == "anthropic":
            message = await client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            # Take the first text block (a response may lead with a non-text block).
            text = next(
                (b.text for b in message.content if getattr(b, "type", None) == "text"),
                "",
            )
        else:
            resp = await _call_openai_chat(
                client,
                model_id,
                max_tokens,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            input_tokens = resp.usage.prompt_tokens
            output_tokens = resp.usage.completion_tokens
            text = resp.choices[0].message.content or ""

        duration = time.perf_counter() - t0
        cost = compute_cost(model_id, input_tokens, output_tokens)

        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)
        span.set_attribute("llm.cost_usd", cost)
        span.set_attribute("llm.duration_seconds", duration)

        log.info(
            "llm_call_complete",
            agent=agent_name,
            provider=provider,
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            duration_seconds=round(duration, 3),
        )

        return text, input_tokens, output_tokens, cost


# ---------------------------------------------------------------------------
# Structured output (prompt-instructed JSON — provider-agnostic)
# ---------------------------------------------------------------------------
async def call_llm_structured(
    *,
    system: str,
    user: str,
    output_schema: type[T],
    model: str = REASON_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    agent_name: str = "unknown",
    token_budget_remaining: int | None = None,
) -> tuple[T, int, int, float]:
    """
    Call the LLM and parse the response as a Pydantic model by instructing it to
    return only JSON conforming to the schema. Works across all providers.
    """
    schema_str = output_schema.model_json_schema()
    structured_system = f"""{system}

IMPORTANT: You MUST respond with ONLY valid JSON that conforms to this schema:
{json.dumps(schema_str, indent=2)}

Do not include any text before or after the JSON. Do not use markdown code blocks."""

    text, inp, out, cost = await call_llm(
        system=structured_system,
        user=user,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        agent_name=agent_name,
        token_budget_remaining=token_budget_remaining,
    )

    # extract_json tolerates fences and trailing prose; the schema still decides
    # whether what came back is usable.
    parsed = output_schema.model_validate(extract_json(text))
    return parsed, inp, out, cost
