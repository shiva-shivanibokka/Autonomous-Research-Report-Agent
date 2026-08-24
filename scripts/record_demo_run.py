"""
Record one real research run and commit it as the replay the hosted demo plays.

Why this exists: the pipeline needs a server that will hold a request open for
several minutes, launch headless Chromium, and keep scraped pages in memory. No
free tier carries that, and a cold start long enough to matter reads as "broken"
to anyone clicking a portfolio link. So instead of hosting a backend, we run the
real thing once — real Tavily searches, real scraping, real LLM calls — and
commit what came out. The deployed UI replays it.

What makes this honest rather than a mockup:
  * Every agent message, token count and dollar figure is what actually happened.
  * The report is the model's own output, not written for the demo.
  * The recording carries its own timings, and the UI states plainly that it is
    a recording, when it was made, and how long the real run took.

Usage:
    python scripts/record_demo_run.py                 # default query
    python scripts/record_demo_run.py --query "..."   # your own
    python scripts/record_demo_run.py --dry-run       # run, print, write nothing

Requires TAVILY_API_KEY and ANTHROPIC_API_KEY in .env. The key is read here and
passed straight to the pipeline; nothing is written to the recording that was
not first checked for it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import re
import sys
import time
from datetime import UTC, datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT = REPO_ROOT / "frontend" / "public" / "demo" / "run.json"

# A recording is a file in a public repo. Anything key-shaped in it is a leak
# that survives forever in git history, so the write is gated on this.
SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{30,}"),
]

# Cap the committed artifact. A recording is a page load, not a download.
MAX_BYTES = 4 * 1024 * 1024

# Chosen because the answer is genuinely contested: controlled trials and
# self-reported surveys disagree about whether AI coding assistants speed
# developers up. That exercises the part of this pipeline worth showing —
# triangulation, confidence scoring and the contradiction map — rather than a
# question every source agrees on, which any single search could answer.
DEFAULT_QUERY = (
    "Do AI coding assistants actually make software developers more productive?"
)


def load_env() -> None:
    """
    Load .env, letting the file win over the ambient environment.

    Deliberately not setdefault. A stale ANTHROPIC_API_KEY exported in a shell
    silently outranks the file the user just edited, and the only symptom is a
    401 that looks like a bad key rather than the wrong key — the exact failure
    this script hit on its first run. For a recorder whose entire job is "use
    the credentials in .env", the file is the source of truth.

    Shadowing is announced rather than performed quietly, because a value being
    replaced under you is worth knowing about either way.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        sys.exit("No .env found. Copy .env.example and add your keys.")

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not value:
            continue
        existing = os.environ.get(key)
        if existing and existing != value:
            print(
                f"  note: {key} was already set in the environment to a "
                f"different value; using the one from .env."
            )
        os.environ[key] = value


def collect_secrets() -> list[str]:
    """Live credential values, so the scan can catch them verbatim as well as by shape."""
    found = []
    for name in (
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
    ):
        value = os.environ.get(name, "").strip()
        if len(value) > 8:
            found.append(value)
    return found


def assert_clean(payload: str, secrets: list[str]) -> None:
    for secret in secrets:
        if secret in payload:
            sys.exit("REFUSING TO WRITE: a live API key appears in the recording.")
    for pattern in SECRET_PATTERNS:
        match = pattern.search(payload)
        if match:
            sys.exit(
                f"REFUSING TO WRITE: key-shaped string in the recording: "
                f"{match.group()[:12]}..."
            )


async def record(query: str, mode: str, rounds: int, budget: int, dry_run: bool) -> int:
    from agents.graph import run_pipeline
    from agents.report_format import render_report_markdown
    from agents.schemas import ReportMode, ResearchState

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY is empty in .env — needed to record.")
    if not os.environ.get("TAVILY_API_KEY", "").strip():
        sys.exit("TAVILY_API_KEY is empty in .env — the search agents need it.")

    model = "claude-sonnet-4-5"
    print(f"Query   : {query}")
    print(f"Mode    : {mode}   rounds<={rounds}   budget={budget:,} tokens")
    print(f"Model   : {model}")
    print("Running the real pipeline. Several minutes; live web + live model.\n")

    started = time.perf_counter()
    frames: list[dict] = []
    seen = 0

    async def on_progress(snapshot) -> None:
        """One frame per new activity entry, stamped with when it really happened."""
        nonlocal seen
        entries = snapshot.activity_log
        while seen < len(entries):
            entry = entries[seen]
            seen += 1
            elapsed = round(time.perf_counter() - started, 2)
            frames.append(
                {
                    "at": elapsed,
                    "entry": entry.model_dump(mode="json"),
                    "round": snapshot.current_round,
                    "tokens_used": snapshot.tokens_used_total,
                    "cost_usd": round(snapshot.cost_usd_total, 6),
                }
            )
            print(
                f"  [{elapsed:7.2f}s] {entry.status.value:9s} "
                f"{entry.agent_name:22s} {entry.message[:70]}"
            )

    state = ResearchState(
        query=query,
        report_mode=ReportMode(mode),
        max_rounds=rounds,
        token_budget=budget,
        provider="anthropic",
        model=model,
        api_key=api_key,
    )

    final = await run_pipeline(state, on_progress=on_progress)
    await on_progress(final)  # catch entries added by the terminal node
    duration = round(time.perf_counter() - started, 2)

    if final.fatal_error:
        print(f"\nRun failed: {final.fatal_error}")
        return 1
    if not final.final_report:
        print("\nRun produced no report; not recording.")
        return 1

    report = final.final_report
    markdown = render_report_markdown(report, final.report_mode.value)

    recording = {
        "schema_version": "1.0",
        "recorded_at": datetime.now(UTC).isoformat(),
        "query": query,
        "report_mode": final.report_mode.value,
        "provider": "anthropic",
        "model": model,
        "duration_seconds": duration,
        "tokens_used": final.tokens_used_total,
        "cost_usd": round(final.cost_usd_total, 6),
        "tokens_by_agent": final.tokens_by_agent,
        "rounds_run": final.current_round + 1,
        "max_rounds": final.max_rounds,
        "converged": final.converged,
        "sources_consulted": len(final.all_sources),
        "frames": frames,
        "activity_log": [e.model_dump(mode="json") for e in final.activity_log],
        "report": report,
        "report_markdown": markdown,
        "quality": report.get("quality", {}),
    }

    payload = json.dumps(recording, indent=2, ensure_ascii=False)

    quality = recording["quality"]
    print("\n" + "=" * 68)
    print(f"  duration        {duration}s")
    print(f"  rounds          {recording['rounds_run']} of {final.max_rounds}")
    print(f"  tokens          {final.tokens_used_total:,}")
    print(f"  cost            ${final.cost_usd_total:.4f}")
    print(f"  claims          {quality.get('total_claims_extracted', 0)}")
    print(f"  citations       {len(report.get('citations', []))}")
    print(f"  contradictions  {len(report.get('contradictions', []))}")
    print(f"  converged       {recording['converged']}")
    print(f"  frames          {len(frames)}")
    print(f"  payload         {len(payload.encode()):,} bytes")
    print("=" * 68)

    # Guard rails before anything touches the working tree.
    assert_clean(payload, collect_secrets())
    if len(payload.encode()) > MAX_BYTES:
        sys.exit(f"REFUSING TO WRITE: recording exceeds {MAX_BYTES:,} bytes.")
    if not report.get("citations"):
        sys.exit("REFUSING TO WRITE: report has no citations — not worth showing.")
    if quality.get("total_claims_extracted", 0) == 0:
        sys.exit("REFUSING TO WRITE: no claims extracted — the run is not healthy.")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(payload, encoding="utf-8", newline="\n")
    print(f"\nWrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--mode", default="general")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--budget", type=int, default=120_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env()
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    return asyncio.run(
        record(args.query, args.mode, args.rounds, args.budget, args.dry_run)
    )


if __name__ == "__main__":
    raise SystemExit(main())
