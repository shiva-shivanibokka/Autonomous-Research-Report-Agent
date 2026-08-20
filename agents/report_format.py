"""
Render a structured report dict (any of the four report modes) into Markdown.

Single source of truth for report formatting — used by the API so the frontend
can render one Markdown string instead of re-implementing four mode-specific
layouts in TypeScript.
"""

from __future__ import annotations


def render_report_markdown(report: dict, mode: str) -> str:
    """Turn a report dict + its mode into a Markdown document."""
    if not report:
        return "_Report not available._"

    lines: list[str] = [f"# {report.get('title', 'Research Report')}\n"]

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
        for work in report.get("related_works", []):
            title = work.get("title", "Untitled")
            authors = work.get("authors", "")
            year = work.get("year", "")
            summary = work.get("summary", "")
            lines.append(f"### {title} ({authors}, {year})\n{summary}\n")
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
        lines.append(
            "\n## Future Directions\n"
            + "\n".join(f"- {d}" for d in report.get("future_directions", []))
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

    # Contradiction map — shared across modes
    contradictions = report.get("contradictions", [])
    if contradictions:
        lines.append("\n## Contradiction Map")
        for c in contradictions:
            lines.append(f"\n**{c.get('topic', '')}**")
            lines.append(f"- Source A: {c.get('source_a_claim', '')}")
            lines.append(f"- Source B: {c.get('source_b_claim', '')}")
            lines.append(f"- Resolution: *{c.get('resolution', '')}*")

    # Citations — shared across modes
    citations = report.get("citations", [])
    if citations:
        lines.append(f"\n## Citations ({len(citations)} sources)")
        for c in citations[:30]:
            title = c.get("title", c.get("url", ""))
            lines.append(
                f"[{c.get('index', '')}] [{title}]({c.get('url', '')}) — "
                f"{c.get('domain', '')} ({c.get('accessed_date', '')})"
            )

    return "\n".join(lines)
