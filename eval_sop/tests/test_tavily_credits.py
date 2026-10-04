"""Tavily credit accounting under concurrency and failure (round-2 review)."""

import asyncio

import pytest

from eval_sop import common


class SlowTavily:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    async def search(self, **kw):
        self.calls += 1
        await asyncio.sleep(0.02)
        if self.fail:
            raise RuntimeError("tavily 500")
        return {"results": [{"url": "https://a.example/x", "title": "t", "content": "c", "score": 0.5}]}


@pytest.fixture
def tavily(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "STATE_DIR", tmp_path / "state")  # fresh credit ledger
    patched = common.install()
    import agents.tools.search_tool as st

    common._tavily_cache.db.execute("DELETE FROM kv")
    monkeypatch.setattr(common, "TAVILY_CREDIT_CAP", 2)
    return patched, st, monkeypatch


def test_concurrent_searches_cannot_overshoot_the_credit_cap(tavily):
    patched, st, mp = tavily
    fake = SlowTavily()
    mp.setattr(st, "_client", fake)

    async def go():
        return await asyncio.gather(*[patched.search(f"q{i}") for i in range(5)],
                                    return_exceptions=True)

    asyncio.run(go())
    assert fake.calls <= 2 and common.credits_spent() <= 2


def test_failed_searches_still_count_against_the_cap(tavily):
    patched, st, mp = tavily
    mp.setattr(st, "_client", SlowTavily(fail=True))
    with pytest.raises(RuntimeError):
        asyncio.run(patched.search("q-fail"))
    assert common.credits_spent() == 1


def test_install_is_idempotent_and_does_not_stack_wrappers(tavily):
    patched, st, mp = tavily
    fake = SlowTavily()
    mp.setattr(st, "_client", fake)
    common.install()
    common.install()
    asyncio.run(st.tavily_search("q-once"))
    assert fake.calls == 1 and common.credits_spent() == 1
