"""Unit tests for report -> Markdown rendering."""

from agents.report_format import render_report_markdown


def test_empty_report():
    assert "not available" in render_report_markdown({}, "general").lower()


def test_general_report_sections_and_citations():
    report = {
        "title": "State of AI Agents 2026",
        "executive_summary": "Agents are everywhere.",
        "key_findings": ["Finding one", "Finding two"],
        "detailed_sections": {"Adoption": "Rising fast."},
        "confidence_assessment": "High overall.",
        "limitations": "Web-sourced only.",
        "citations": [
            {
                "index": 1,
                "title": "Example",
                "url": "https://example.com",
                "domain": "example.com",
                "accessed_date": "2026-07-12",
            }
        ],
    }
    md = render_report_markdown(report, "general")
    assert "# State of AI Agents 2026" in md
    assert "## Key Findings" in md
    assert "- Finding one" in md
    assert "## Adoption" in md  # dynamic detailed section
    assert "https://example.com" in md
    assert "Citations (1 sources)" in md


def test_competitive_report_lists_competitors():
    report = {
        "title": "CI Report",
        "executive_summary": "s",
        "market_overview": "o",
        "competitors": [
            {
                "name": "Acme",
                "market_position": "leader",
                "strengths": ["fast"],
                "weaknesses": ["pricey"],
            }
        ],
        "competitive_matrix_summary": "m",
        "whitespace_opportunities": ["gap"],
        "strategic_recommendations": ["do x"],
    }
    md = render_report_markdown(report, "competitive_intelligence")
    assert "### Acme" in md
    assert "Whitespace Opportunities" in md


def test_contradiction_map_rendered_when_present():
    report = {
        "title": "T",
        "executive_summary": "s",
        "key_findings": [],
        "detailed_sections": {},
        "confidence_assessment": "",
        "limitations": "",
        "citations": [],
        "contradictions": [
            {
                "topic": "Market size",
                "source_a_claim": "$10B",
                "source_b_claim": "$20B",
                "resolution": "range",
            }
        ],
    }
    md = render_report_markdown(report, "general")
    assert "Contradiction Map" in md
    assert "Market size" in md
