"""Failed page fetches must not be cached forever (round-2 review)."""

import asyncio

from eval_sop import common


def test_failed_fetch_is_retried_in_a_later_process(monkeypatch):
    from agents.schemas import ScrapedPage

    calls = {"n": 0}

    async def flaky(url, title=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return ScrapedPage(url=url, title=title, content="", word_count=0, scrape_error="403")
        return ScrapedPage(url=url, title=title, content="real text", word_count=2)

    saved = common._ORIGINALS.get("scrape_page")
    common._ORIGINALS["scrape_page"] = flaky
    try:
        patched = common.install()
        url = "https://flaky.example/page"
        patched.pages.db.execute("DELETE FROM kv WHERE k=?", (url,))
        first = asyncio.run(patched.scrape(url))
        again = asyncio.run(patched.scrape(url))  # same process: reuse, keeps c/d inputs identical
        assert first.scrape_error and again.scrape_error and calls["n"] == 1
        assert patched.pages.get(url)["failed_in_pid"]  # failure is marked, not silent
        monkeypatch.setattr(common.os, "getpid", lambda: -12345)  # a later process
        later = asyncio.run(patched.scrape(url))
        assert later.content == "real text" and calls["n"] == 2
    finally:
        if saved is None:
            common._ORIGINALS.pop("scrape_page", None)
        else:
            common._ORIGINALS["scrape_page"] = saved
        common.install()
