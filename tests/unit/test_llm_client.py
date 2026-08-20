"""Unit tests for the multi-provider LLM client — no network calls."""

import anthropic
import openai
import pytest

from agents import llm_client
from agents.llm_client import (
    SUPPORTED_PROVIDERS,
    _build_client,
    compute_cost,
    set_llm_creds,
)


def test_compute_cost_exact_match():
    # Haiku 4.5: $1.00 in / $5.00 out per 1M tokens.
    cost = compute_cost("claude-haiku-4-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(6.00)


def test_compute_cost_prefix_match():
    # A dated/suffixed id should match its family by prefix.
    assert compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(
        1.00
    )


def test_compute_cost_unknown_model_uses_default():
    # Unknown model → default estimate ($1 in / $3 out), never crashes.
    assert compute_cost("some-future-model", 1_000_000, 1_000_000) == pytest.approx(
        4.00
    )


def test_build_client_anthropic():
    client = _build_client("anthropic", "sk-test")
    assert isinstance(client, anthropic.AsyncAnthropic)


@pytest.mark.parametrize("provider", ["openai", "groq", "google"])
def test_build_client_openai_compatible(provider):
    client = _build_client(provider, "sk-test")
    assert isinstance(client, openai.AsyncOpenAI)


def test_build_client_rejects_unknown_provider():
    with pytest.raises(ValueError):
        _build_client("nope", "sk-test")


def test_all_supported_providers_build():
    for p in SUPPORTED_PROVIDERS:
        assert _build_client(p, "sk-test") is not None


def test_set_llm_creds_populates_contextvar():
    set_llm_creds("groq", "llama-3.3-70b-versatile", "sk-test")
    creds = llm_client._creds_var.get()
    assert creds is not None
    assert creds.provider == "groq"
    assert creds.model == "llama-3.3-70b-versatile"
