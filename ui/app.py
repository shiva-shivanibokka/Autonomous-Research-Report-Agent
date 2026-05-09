"""
Gradio UI for the Autonomous Research Report Agent.

Features:
- Research query input with mode selector
- Live agent activity feed (polls /report/status every 2s)
- Quality score dashboard
- Full report display with citations and contradiction map
- Cost and token accounting per report
- Recent reports history tab
"""

from __future__ import annotations

import json
import os
import time

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")
POLL_INTERVAL = 2.5  # seconds between status polls


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _post_generate(query: str, mode: str, max_rounds: int, token_budget: int) -> dict:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{API_URL}/report/generate",
            json={
                "query": query,
                "report_mode": mode,
                "max_rounds": max_rounds,
                "token_budget": token_budget,
            },
        )
        resp.raise_for_status()
        return resp.json()


def _get_status(job_id: str) -> dict:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/report/status/{job_id}")
        resp.raise_for_status()
        return resp.json()


def _get_report(job_id: str) -> dict:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/report/{job_id}")
        resp.raise_for_status()
        return resp.json()


def _get_recent() -> list[dict]:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/reports?limit=10")
        resp.raise_for_status()
        return resp.json().get("reports", [])


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "pending": "⏳",
    "running": "🔄",
    "completed": "✅",
    "failed": "❌",
    "skipped": "⏭️",
}

_MODE_MAP = {
    "General Research": "general",
    "Competitive Intelligence": "competitive_intelligence",
    "Investment Thesis": "investment_thesis",
    "Academic Literature Review": "academic_literature_review",
}


def _format_activity_log(activity_log: list[dict]) -> str:
    if not activity_log:
        return "_No activity yet..._"
    lines = []
    for entry in activity_log:
        icon = _STATUS_ICONS.get(entry.get("status", ""), "•")
        agent = entry.get("agent_name", "?")
        msg = entry.get("message", "")
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        tokens = entry.get("tokens_used", 0)
        cost = entry.get("cost_usd", 0.0)
        suffix = f" | {tokens:,} tokens | ${cost:.4f}" if tokens else ""
        lines.append(f"`{ts}` {icon} **{agent}**: {msg}{suffix}")
    return "\n\n".join(lines)


def _format_quality(quality: dict) -> str:
    if not quality:
        return ""
    return f"""
### Quality Report
| Metric | Score |
|--------|-------|
| Overall Quality | {quality.get("overall_quality_score", 0):.0%} |
| Coverage | {quality.get("coverage_score", 0):.0%} |
| Source Diversity | {quality.get("source_diversity_score", 0):.0%} |
| Contradiction Rate | {quality.get("contradiction_rate", 0):.0%} |

**Claims:** {quality.get("total_claims_extracted", 0)} extracted | {quality.get("claims_flagged_by_critic", 0)} flagged | {quality.get("claims_verified_by_fact_checker", 0)} verified

**Confidence Distribution:**
- HIGH: {quality.get("confidence_distribution", {}).get("high", 0)}
- MEDIUM: {quality.get("confidence_distribution", {}).get("medium", 0)}
- LOW: {quality.get("confidence_distribution", {}).get("low", 0)}
- CONTESTED: {quality.get("confidence_distribution", {}).get("contested", 0)}

**{quality.get("convergence_note", "")}**
"""


def _format_report_body(report: dict, mode: str) -> str:
    if not report:
        return "_Report not available._"

    lines = [f"# {report.get('title', 'Research Report')}\n"]

    if mode == "competitive_intelligence":
        lines.append(f"## Executive Summary\n{report.get('executive_summary', '')}\n")
        lines.append(f"## Market Overview\n{report.get('market_overview', '')}\n")
        for comp in report.get("competitors", []):
            lines.append(f"### {comp.get('name', 'Competitor')}")
            lines.append(f"**Position:** {comp.get('market_position', '')}")
            lines.append(f"**Strengths:** {', '.join(comp.get('strengths', []))}")
            lines.append(f"**Weaknesses:** {', '.join(comp.get('weaknesses', []))}\n")
        lines.append(
            f"## Competitive Matrix\n{report.get('competitive_matrix_summary', '')}\n"
        )
        lines.append(
            "## Whitespace Opportunities\n"
            + "\n".join(f"- {w}" for w in report.get("whitespace_opportunities", []))
        )
        lines.append(
            "\n## Strategic Recommendations\n"
            + "\n".join(f"- {r}" for r in report.get("strategic_recommendations", []))
        )

    elif mode == "investment_thesis":
        lines.append(f"## Executive Summary\n{report.get('executive_summary', '')}\n")
        lines.append(f"## Thesis\n{report.get('thesis_statement', '')}\n")
        sizing = report.get("market_sizing", {})
        if sizing:
            lines.append(
                "## Market Sizing\n"
                + "\n".join(f"- **{k}:** {v}" for k, v in sizing.items())
            )
        lines.append(
            "\n## Bull Case\n"
            + "\n".join(f"- {b}" for b in report.get("bull_case", []))
        )
        lines.append(
            "\n## Bear Case\n"
            + "\n".join(f"- {b}" for b in report.get("bear_case", []))
        )
        lines.append(
            "\n## Risk Factors\n"
            + "\n".join(f"- {r}" for r in report.get("risk_factors", []))
        )

    elif mode == "academic_literature_review":
        lines.append(f"## Abstract\n{report.get('abstract', '')}\n")
        lines.append(f"## Introduction\n{report.get('introduction', '')}\n")
        lines.append(
            f"## Methodology Comparison\n{report.get('methodology_comparison', '')}\n"
        )
        lines.append(
            f"## Key Findings Synthesis\n{report.get('key_findings_synthesis', '')}\n"
        )
        lines.append(
            "\n## Research Gaps\n"
            + "\n".join(f"- {g}" for g in report.get("research_gaps", []))
        )

    else:  # general
        lines.append(f"## Executive Summary\n{report.get('executive_summary', '')}\n")
        lines.append(
            "## Key Findings\n"
            + "\n".join(f"- {f}" for f in report.get("key_findings", []))
        )
        for section, content in report.get("detailed_sections", {}).items():
            lines.append(f"\n## {section}\n{content}")
        lines.append(
            f"\n## Confidence Assessment\n{report.get('confidence_assessment', '')}"
        )
        lines.append(f"\n## Limitations\n{report.get('limitations', '')}")

    # Contradiction map
    contradictions = report.get("contradictions", [])
    if contradictions:
        lines.append("\n## Contradiction Map")
        for c in contradictions:
            lines.append(f"\n**{c.get('topic', '')}**")
            lines.append(f"- Source A: {c.get('source_a_claim', '')}")
            lines.append(f"- Source B: {c.get('source_b_claim', '')}")
            lines.append(f"- Resolution: *{c.get('resolution', '')}*")

    # Citations
    citations = report.get("citations", [])
    if citations:
        lines.append(f"\n## Citations ({len(citations)} sources)")
        for c in citations[:20]:
            lines.append(
                f"[{c.get('index', '')}] [{c.get('title', c.get('url', ''))}]({c.get('url', '')}) — {c.get('domain', '')} ({c.get('accessed_date', '')})"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generation flow (generator for streaming updates)
# ---------------------------------------------------------------------------


def generate_report(query: str, mode_label: str, max_rounds: int, token_budget_k: int):
    """
    Generator function that yields UI updates as the report is generated.
    Gradio's gr.update() mechanism polls this generator every iteration.
    """
    if not query or len(query.strip()) < 10:
        yield (
            gr.update(value="Please enter a research question (min 10 characters)."),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
        )
        return

    mode = _MODE_MAP.get(mode_label, "general")
    token_budget = token_budget_k * 1000

    # Submit job
    try:
        job_response = _post_generate(query, mode, max_rounds, token_budget)
        job_id = job_response["job_id"]
    except Exception as e:
        yield (
            gr.update(value=f"Failed to submit job: {str(e)}"),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
        )
        return

    # Poll for status
    while True:
        try:
            status = _get_status(job_id)
        except Exception as e:
            yield (
                gr.update(value=f"Status poll failed: {str(e)}"),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
            )
            time.sleep(POLL_INTERVAL)
            continue

        activity_md = _format_activity_log(status.get("activity_log", []))
        cost_info = f"**Tokens:** {status.get('tokens_used', 0):,} | **Cost:** ${status.get('cost_usd', 0):.4f} | **Round:** {status.get('current_round', 0) + 1}"

        if status["status"] == "completed":
            try:
                report_data = _get_report(job_id)
                report_md = _format_report_body(report_data.get("report", {}), mode)
                quality_md = _format_quality(
                    report_data.get("report", {}).get("quality", {})
                )
                duration = report_data.get("duration_seconds", 0)
                cost_info += f" | **Duration:** {duration:.1f}s"
            except Exception as e:
                report_md = f"Error fetching report: {str(e)}"
                quality_md = ""

            yield (
                gr.update(value=activity_md),
                gr.update(value=cost_info),
                gr.update(value=quality_md),
                gr.update(value=report_md),
            )
            return

        elif status["status"] == "failed":
            yield (
                gr.update(
                    value=activity_md + f"\n\n**FAILED:** {status.get('error', '')}"
                ),
                gr.update(value=cost_info),
                gr.update(value=""),
                gr.update(value=""),
            )
            return

        else:
            yield (
                gr.update(value=activity_md),
                gr.update(value=cost_info),
                gr.update(value=""),
                gr.update(value=""),
            )
            time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui():
    with gr.Blocks(
        title="Autonomous Research Report Agent",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
# Autonomous Research Report Agent
Multi-agent AI pipeline that produces cited, quality-scored research reports.

**Architecture:** Orchestrator → Parallel Search Agents → Scraper Agents → Analyst Agents → Critic (self-improving loop) → Fact-Checkers → Writer

**Features:** Source triangulation · Contradiction detection · Per-report cost tracking · 4 report modes
        """)

        with gr.Tab("Generate Report"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Research Question",
                        placeholder="e.g. What is the competitive landscape for AI coding assistants in 2025?",
                        lines=3,
                    )
                with gr.Column(scale=1):
                    mode_input = gr.Dropdown(
                        label="Report Mode",
                        choices=list(_MODE_MAP.keys()),
                        value="General Research",
                    )
                    rounds_input = gr.Slider(
                        label="Max Research Rounds (self-improving loops)",
                        minimum=1,
                        maximum=3,
                        step=1,
                        value=2,
                    )
                    budget_input = gr.Slider(
                        label="Token Budget (thousands)",
                        minimum=20,
                        maximum=200,
                        step=10,
                        value=80,
                    )

            generate_btn = gr.Button("Generate Report", variant="primary", size="lg")

            cost_display = gr.Markdown(label="Cost & Token Accounting")

            with gr.Row():
                with gr.Column(scale=1):
                    activity_display = gr.Markdown(
                        label="Live Agent Activity Feed",
                        value="_Submit a query to see live agent updates..._",
                    )
                with gr.Column(scale=1):
                    quality_display = gr.Markdown(
                        label="Report Quality Metrics",
                        value="",
                    )

            report_display = gr.Markdown(
                label="Generated Report",
                value="",
            )

            generate_btn.click(
                fn=generate_report,
                inputs=[query_input, mode_input, rounds_input, budget_input],
                outputs=[
                    activity_display,
                    cost_display,
                    quality_display,
                    report_display,
                ],
            )

        with gr.Tab("Recent Reports"):
            refresh_btn = gr.Button("Refresh", variant="secondary")
            recent_display = gr.DataFrame(
                headers=[
                    "Job ID",
                    "Status",
                    "Mode",
                    "Query",
                    "Tokens",
                    "Cost ($)",
                    "Duration (s)",
                    "Created",
                ],
                label="Recent Research Jobs",
            )

            def load_recent():
                try:
                    jobs = _get_recent()
                    rows = []
                    for j in jobs:
                        rows.append(
                            [
                                j.get("job_id", "")[:8],
                                j.get("status", ""),
                                j.get("report_mode", ""),
                                j.get("query", "")[:60] + "..."
                                if len(j.get("query", "")) > 60
                                else j.get("query", ""),
                                j.get("tokens_used", 0),
                                round(float(j.get("cost_usd", 0)), 4),
                                round(float(j.get("duration_seconds", 0) or 0), 1),
                                str(j.get("created_at", ""))[:19],
                            ]
                        )
                    return rows
                except Exception as e:
                    return [[f"Error: {str(e)}", "", "", "", "", "", "", ""]]

            refresh_btn.click(fn=load_recent, outputs=recent_display)
            demo.load(fn=load_recent, outputs=recent_display)

        with gr.Tab("Architecture"):
            gr.Markdown("""
## Agent Pipeline Architecture

```
User Query
    │
    ▼
Orchestrator Agent
(decomposes query into 3-5 sub-questions, allocates token budgets)
    │
    ▼ [parallel asyncio.gather()]
┌───────────────────────────────────┐
│  Search Agent 1  │  Search Agent 2  │  ...  │  Search Agent N  │
│  (Tavily search) │  (Tavily search) │       │  (Tavily search) │
└───────────────────────────────────┘
    │
    ▼ [parallel scraping]
┌───────────────────────────────────┐
│  Scraper Agent 1 │  Scraper Agent 2 │  ...  │ (Playwright fallback) │
└───────────────────────────────────┘
    │
    ▼ [parallel analysis]
┌───────────────────────────────────┐
│  Analyst Agent 1 │  Analyst Agent 2 │  ...  │
│  (source triangulation, claim extraction) │
└───────────────────────────────────┘
    │
    ▼
Critic Agent
(quality scoring, flags low-confidence claims)
    │
    ├──[needs_more_research=True]──→ increment_round → Orchestrator (loop up to max_rounds)
    │
    ▼ [converged]
Fact-Checker Agents (spawned per flagged claim)
    │
    ▼
Writer Agent
(synthesizes into domain-specific structured report)
    │
    ▼
Final Report + Quality Score + Contradiction Map + Citations
```

## Infrastructure
- **FastAPI** backend with async endpoints
- **Celery + Redis** async job queue (non-blocking UI)
- **PostgreSQL** persistent job storage
- **Prometheus + Grafana** metrics and dashboards
- **Jaeger (OTLP)** distributed tracing across all agents
- **Alertmanager + Slack** configured alert rules
- **GitHub Actions** CI/CD (lint, test, build, deploy)
            """)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
