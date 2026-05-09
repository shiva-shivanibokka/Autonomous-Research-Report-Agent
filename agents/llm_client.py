"""
Centralized LLM client with:
- Token counting and cost tracking per call
- Structured output via instructor
- Retry with exponential backoff (tenacity)
- OpenTelemetry span per call
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

import anthropic
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
# Token cost table (per 1M tokens, USD) — Claude models as of 2025
# ---------------------------------------------------------------------------
_COST_TABLE: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
}

# Model aliases used throughout agents
FAST_MODEL = "claude-haiku-4-5"  # extraction, parsing (cheap + fast)
REASON_MODEL = "claude-sonnet-4-5"  # reasoning, analysis, writing (balanced)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return cost in USD for a given LLM call."""
    prices = _COST_TABLE.get(model, {"input": 3.00, "output": 15.00})
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]


# ---------------------------------------------------------------------------
# Raw text completion (for agents that just need a string back)
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.APIConnectionError)
    ),
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
    temperature: float = 0.2,
    agent_name: str = "unknown",
    token_budget_remaining: int | None = None,
) -> tuple[str, int, int, float]:
    """
    Call the LLM and return (response_text, input_tokens, output_tokens, cost_usd).

    Args:
        system: System prompt
        user: User message
        model: Claude model ID
        max_tokens: Max output tokens. Capped at token_budget_remaining if provided.
        temperature: Sampling temperature
        agent_name: Which agent is calling (for logging/tracing)
        token_budget_remaining: If set, caps max_tokens to avoid budget overrun

    Returns:
        (text, input_tokens, output_tokens, cost_usd)
    """
    if token_budget_remaining is not None:
        max_tokens = min(max_tokens, max(256, token_budget_remaining))

    with tracer.start_as_current_span(f"llm.{agent_name}") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.agent", agent_name)
        span.set_attribute("llm.max_tokens", max_tokens)

        t0 = time.perf_counter()
        client = anthropic.AsyncAnthropic()

        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        duration = time.perf_counter() - t0
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost = compute_cost(model, input_tokens, output_tokens)
        text = message.content[0].text

        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)
        span.set_attribute("llm.cost_usd", cost)
        span.set_attribute("llm.duration_seconds", duration)

        log.info(
            "llm_call_complete",
            agent=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            duration_seconds=round(duration, 3),
        )

        return text, input_tokens, output_tokens, cost


# ---------------------------------------------------------------------------
# Structured output completion (returns a validated Pydantic model)
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
    Call the LLM and parse the response as a Pydantic model.
    Uses Claude's native JSON mode by instructing it to return valid JSON.

    Returns:
        (parsed_model, input_tokens, output_tokens, cost_usd)
    """
    import json

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

    # Strip any accidental markdown fencing
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    parsed = output_schema.model_validate_json(text)
    return parsed, inp, out, cost
