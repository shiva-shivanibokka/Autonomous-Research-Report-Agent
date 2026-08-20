"""
Web scraper with Playwright fallback for JS-rendered pages.
Returns cleaned text content with source attribution.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import time
from urllib.parse import urlparse

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

# Hard ceiling on bytes downloaded per page. MAX_CONTENT_CHARS truncates the
# *cleaned text*, which happens only after the whole body is already in memory —
# so a single large response could exhaust the container before it applied.
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024


class BlockedURLError(Exception):
    """Raised when a URL resolves somewhere a scraper has no business going."""


async def _assert_public_url(url: str) -> None:
    """
    Refuse non-HTTP schemes and hosts that resolve to private address space.

    These URLs come from Tavily and from redirects, i.e. entirely from outside.
    On a cloud host the loopback and link-local ranges are where the metadata
    service and any sidecar admin ports live, so an unrestricted fetcher that
    follows redirects is an SSRF primitive pointed at its own infrastructure.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedURLError(f"unsupported scheme: {parsed.scheme or 'none'}")
    host = parsed.hostname
    if not host:
        raise BlockedURLError("no host in URL")

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise BlockedURLError(f"could not resolve {host}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise BlockedURLError(f"{host} resolves to non-public address {ip}")


def _extract_title(html: str, fallback: str) -> str:
    """Read <title> out of a document, falling back to the URL."""
    title_tag = BeautifulSoup(html, "html.parser").find("title")
    return title_tag.get_text().strip() if title_tag else fallback


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
    """
    Lightweight HTTP fetch using httpx, with a size cap and per-hop URL checks.

    Redirects are followed manually rather than by httpx, because
    `follow_redirects=True` checks nothing between hops — a page that 302s to a
    link-local address would sail straight past a guard applied only to the
    URL we started with.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        follow_redirects=False,
        headers=headers,
    ) as client:
        for _ in range(5):
            await _assert_public_url(url)
            async with client.stream("GET", url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        response.raise_for_status()
                        raise BlockedURLError("redirect without a Location header")
                    url = str(response.url.join(location))
                    continue

                response.raise_for_status()

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        log.warning("scrape_truncated_oversize", url=url[:80])
                        break
                    chunks.append(chunk)

                body = b"".join(chunks)
                encoding = response.encoding or "utf-8"
                return body.decode(encoding, errors="replace")

        raise BlockedURLError("too many redirects")


async def _fetch_with_playwright(url: str) -> str:
    """Playwright fallback for JS-rendered pages."""
    await _assert_public_url(url)
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright not installed — run: pip install playwright && playwright install chromium"
        ) from exc

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

        content = (await asyncio.to_thread(_clean_html, html))[:MAX_CONTENT_CHARS]
        word_count = len(content.split())
        duration = time.perf_counter() - t0

        # Extract title from HTML if not provided
        if not title:
            title = await asyncio.to_thread(_extract_title, html, url)

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
    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded_scrape(url: str, title: str) -> ScrapedPage:
        async with sem:
            return await scrape_page(url, title)

    tasks = [_bounded_scrape(url, title) for url, title in urls_with_titles]
    return await asyncio.gather(*tasks)
