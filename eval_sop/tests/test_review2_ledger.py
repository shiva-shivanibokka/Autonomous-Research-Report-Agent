"""
Round-2 review scenarios (scratchpad/review2/attack.py) as tests: paid calls
that never reached the ledger, CJK under-estimation, concurrency.

Every fake here raises from inside `get_final_message()` — i.e. after the
request was accepted and the stream started — which is where a real stream
fails mid-response and where the call may already be billed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from eval_sop import budget, common

MODEL = "claude-haiku-4-5-20251001"
REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _msg(inp=1000, out=200):
    return SimpleNamespace(
        id="m", model=MODEL, content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


class _MidStream:
    def __init__(self, outcome, delay=0.0):
        self.outcome, self.delay = outcome, delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get_final_message(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class Fake:
    def __init__(self, f, delay=0.0):
        self.f, self.delay, self.calls = f, delay, 0
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kw):
        self.calls += 1
        return _MidStream(self.f(self.calls), self.delay)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    common.install()
    monkeypatch.setattr(common, "BACKEND", "anthropic")
    monkeypatch.setattr(common, "GEN_MODEL", MODEL)
    led = budget.UsdLedger(tmp_path / "l.sqlite", 1.0)
    monkeypatch.setattr(common, "LEDGER", led)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(common, "_sleep", no_sleep)
    common._llm_cache.db.execute("DELETE FROM kv")
    common._llm_cache.db.commit()
    yield led
    common.set_anthropic_client(None)


def _rows(led):
    return list(led.db.execute("SELECT status, cost_usd FROM calls ORDER BY id"))


async def _call(text="hi", max_tokens=100):
    return await common.chat(MODEL, [{"role": "user", "content": text}], max_tokens=max_tokens, agent="x")


def _worst(text="hi", max_tokens=100):
    return budget.worst_call_cost(MODEL, [{"role": "user", "content": text}], max_tokens)


def test_midstream_overloaded_status_200_is_retried_and_charged(ledger):
    err = anthropic.APIStatusError("overloaded", response=httpx.Response(200, request=REQ),
                                   body={"type": "error", "error": {"type": "overloaded_error"}})
    common.set_anthropic_client(Fake(lambda i: err if i == 1 else _msg()))
    assert asyncio.run(_call())[0] == "ok"
    rows = _rows(ledger)
    assert len(rows) == 2
    assert rows[0][1] == pytest.approx(_worst())  # the failed stream is charged at worst case
    assert rows[1][0] == "ok"


def test_midstream_httpx_read_timeout_is_charged_and_bounded(ledger):
    common.set_anthropic_client(Fake(lambda i: httpx.ReadTimeout("read timed out")))
    with pytest.raises(common.TransportError):
        asyncio.run(_call())
    rows = _rows(ledger)
    assert len(rows) == 3  # first try + 2 retries, never more
    assert all(c == pytest.approx(_worst()) for _, c in rows)


def test_midstream_remote_protocol_error_is_charged(ledger):
    common.set_anthropic_client(Fake(lambda i: httpx.RemoteProtocolError("peer closed") if i == 1 else _msg()))
    asyncio.run(_call())
    assert _rows(ledger)[0][1] == pytest.approx(_worst())


def test_keyboard_interrupt_mid_call_leaves_a_worst_case_charge(ledger):
    common.set_anthropic_client(Fake(lambda i: KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(_call())
    assert ledger.spent() == pytest.approx(_worst())


def test_cancelled_call_leaves_a_worst_case_charge(ledger):
    common.set_anthropic_client(Fake(lambda i: _msg(), delay=5))

    async def go():
        t = asyncio.create_task(_call())
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(go())
    assert ledger.spent() == pytest.approx(_worst())


def test_twenty_concurrent_calls_never_exceed_the_cap(ledger):
    ledger.cap = 0.05
    common.set_anthropic_client(Fake(lambda i: _msg(1000, 2000), delay=0.05))

    async def go():
        return await asyncio.gather(*[_call(f"q{i}", 2000) for i in range(20)], return_exceptions=True)

    asyncio.run(go())
    assert ledger.spent() <= ledger.cap + 1e-12


def test_resetting_out_dir_cannot_reset_spend(tmp_path, monkeypatch):
    """The ledger lives at one per-project path, independent of SOP_OUT_DIR."""
    p1 = common.ledger_path()
    monkeypatch.setattr(common, "OUT_DIR", tmp_path / "elsewhere")
    assert common.ledger_path() == p1
