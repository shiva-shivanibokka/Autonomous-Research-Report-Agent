"""Unit tests for Pydantic schemas — validates all agent boundary models."""

import pytest
from pydantic import ValidationError

from agents.schemas import (
    AnalystAgentOutput,
    Claim,
    ConfidenceLevel,
    ReportMode,
    ResearchState,
    SearchAgentOutput,
    SearchResult,
)


def test_research_state_defaults():
    state = ResearchState(query="What is the AI market size in 2025?")
    assert state.query == "What is the AI market size in 2025?"
    assert state.report_mode == ReportMode.GENERAL
    assert state.max_rounds == 3
    assert state.current_round == 0
    assert state.tokens_used_total == 0
    assert len(state.activity_log) == 0


def test_research_state_generates_unique_ids():
    s1 = ResearchState(query="query 1")
    s2 = ResearchState(query="query 2")
    assert s1.job_id != s2.job_id
    assert s1.trace_id != s2.trace_id


def test_claim_confidence_levels():
    claim = Claim(
        text="AI market will reach $500B by 2025",
        source_urls=["https://example.com"],
        confidence=ConfidenceLevel.HIGH,
        supporting_sources=3,
        contradicting_sources=0,
    )
    assert claim.confidence == ConfidenceLevel.HIGH
    assert claim.supporting_sources == 3


def test_search_result_relevance_bounds():
    with pytest.raises(ValidationError):
        SearchResult(
            url="https://example.com",
            title="Test",
            snippet="Test snippet",
            relevance_score=1.5,  # > 1.0 — should fail
        )


def test_search_agent_output_defaults():
    output = SearchAgentOutput(
        sub_question="What is the market size?",
        results=[],
    )
    assert output.tokens_used == 0
    assert output.error is None


def test_analyst_output_with_claims():
    claims = [
        Claim(
            text="The market is growing",
            source_urls=["https://a.com", "https://b.com"],
            confidence=ConfidenceLevel.MEDIUM,
            supporting_sources=2,
        )
    ]
    output = AnalystAgentOutput(
        sub_question="How is the market growing?",
        key_claims=claims,
        contradictions_found=0,
        avg_confidence=0.67,
    )
    assert len(output.key_claims) == 1
    assert output.avg_confidence == 0.67


def test_report_mode_values():
    assert ReportMode.COMPETITIVE_INTELLIGENCE.value == "competitive_intelligence"
    assert ReportMode.INVESTMENT_THESIS.value == "investment_thesis"
    assert ReportMode.ACADEMIC_LITERATURE_REVIEW.value == "academic_literature_review"
    assert ReportMode.GENERAL.value == "general"
