"""
Round-3 review: llm_client.call_llm falls back to a raw anthropic.AsyncAnthropic()
when no per-job creds are set, which would bypass the eval's ledger. After
common.install() that fallback must refuse instead of building a client.
"""

import asyncio
import contextvars

import anthropic
import pytest

from eval_sop import common


def test_raw_client_fallback_cannot_bypass_the_ledger(monkeypatch):
    built = []

    class Recorder:
        def __init__(self, *a, **k):
            built.append(1)
            raise AssertionError("a raw AsyncAnthropic client was constructed")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", Recorder)  # no network either way
    common.install()
    import agents.llm_client as llm_client

    async def call():
        return await llm_client.call_llm(system="s", user="u", agent_name="probe")

    ctx = contextvars.Context()  # fresh context: no per-job creds set
    with pytest.raises(Exception) as e:
        ctx.run(asyncio.run, call())
    assert built == [], "the fallback reached anthropic.AsyncAnthropic()"
    assert "ledger" in str(e.value).lower()
