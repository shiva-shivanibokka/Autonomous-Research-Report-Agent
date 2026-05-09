"""
Web scraper with Playwright fallback for JS-rendered pages.
Returns cleaned text content with source attribution.
"""

from __future__ import annotations

import re
import time

import httpx
import structlog
from bs4 import BeautifulSoup
from opentelemetry import trace
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.schemas import ScrapedPage

log = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Max characters to extract per page to stay within token budgets
MAX_CONTENT_CHARS = 8_000
REQUEST_TIMEOUT = 12  # seconds


def _clean_html(html: str) -> str:
    """Extract and clean readable text from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]
    ):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _fetch_with_httpx(url: str) -> str:
    """Lightweight HTTP fetch using httpx."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def _fetch_with_playwright(url: str) -> str:
    """Playwright fallback for JS-rendered pages."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError(
            "playwright not installed — run: pip install playwright && playwright install chromium"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        content = await page.content()
        await browser.close()
        return content


async def scrape_page(url: str, title: str = "") -> ScrapedPage:
    """
    Scrape a single URL. Tries httpx first, falls back to Playwright.

    Returns a ScrapedPage with cleaned text content.
    """
    with tracer.start_as_current_span("tool.scrape_page") as span:
        span.set_attribute("scraper.url", url)
        t0 = time.perf_counter()
        used_playwright = False

        try:
            html = await _fetch_with_httpx(url)
        except Exception as e_http:
            log.warning(
                "httpx_scrape_failed_trying_playwright", url=url, error=str(e_http)
            )
            try:
                html = await _fetch_with_playwright(url)
                used_playwright = True
            except Exception as e_pw:
                log.error("scrape_failed_both_methods", url=url, error=str(e_pw))
                return ScrapedPage(
                    url=url,
                    title=title,
                    content="",
                    word_count=0,
                    used_playwright=False,
                    scrape_error=str(e_pw)[:200],
                )

        content = _clean_html(html)[:MAX_CONTENT_CHARS]
        word_count = len(content.split())
        duration = time.perf_counter() - t0

        # Extract title from HTML if not provided
        if not title:
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else url

        span.set_attribute("scraper.used_playwright", used_playwright)
        span.set_attribute("scraper.word_count", word_count)
        span.set_attribute("scraper.duration_seconds", duration)

        log.info(
            "page_scraped",
            url=url[:80],
            word_count=word_count,
            used_playwright=used_playwright,
            duration_seconds=round(duration, 3),
        )

        return ScrapedPage(
            url=url,
            title=title[:200],
            content=content,
            word_count=word_count,
            used_playwright=used_playwright,
        )


async def scrape_pages(
    urls_with_titles: list[tuple[str, str]], max_concurrent: int = 4
) -> list[ScrapedPage]:
    """Scrape multiple pages with bounded concurrency."""
    import asyncio
    from asyncio import Semaphore

    sem = Semaphore(max_concurrent)

    async def _bounded_scrape(url: str, title: str) -> ScrapedPage:
        async with sem:
            return await scrape_page(url, title)

    tasks = [_bounded_scrape(url, title) for url, title in urls_with_titles]
    return await asyncio.gather(*tasks)
