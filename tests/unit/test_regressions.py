"""
Regression tests for bugs found by reading the code rather than by running it.

Every test here corresponds to a defect that shipped: the pipeline's own test
suite was green throughout, because each of these lives on a path that only
executes with a live database, a live LLM, or a re-research round.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from agents.critic_agent import CriticStructuredOutput
from agents.llm_client import (
    _max_tokens_param_for,
    compute_cost,
    extract_json,
)
from agents.schemas import (
    Claim,
    ConfidenceLevel,
    CriticAgentOutput,
    ResearchState,
    SearchAgentOutput,
    SearchResult,
)
from agents.tools.search_tool import _clamp_score
from api.database import _prepare_updates


# ---------------------------------------------------------------------------
# The database write that failed on every completed job
# ---------------------------------------------------------------------------
def test_completed_at_iso_string_is_coerced_to_datetime():
    """
    asyncpg's timestamptz codec raises TypeError on a str. Both job runners used
    to pass datetime.now(...).isoformat(), so the final write of every successful
    job threw — and the handler recording the failure threw the same way, leaving
    the job on 'running' forever.
    """
    _, values = _prepare_updates(
        {"completed_at": "2026-08-20T12:00:00+00:00", "status": "completed"}
    )
    assert isinstance(values[0], datetime)
    assert values[1] == "completed"


def test_completed_at_datetime_passes_through():
    _, values = _prepare_updates({"completed_at": datetime.now(UTC)})
    assert isinstance(values[0], datetime)


def test_update_rejects_unknown_column():
    """The SET clause is built by interpolation, so column names must be checked."""
    with pytest.raises(ValueError, match="unknown column"):
        _prepare_updates({"status = 'x'; DROP TABLE research_jobs; --": 1})


def test_update_serialises_json_columns():
    _, values = _prepare_updates({"report": {"title": "t"}})
    assert values[0] == '{"title": "t"}'


# ---------------------------------------------------------------------------
# Citations lost across a re-research round
# ---------------------------------------------------------------------------
def _source(url: str) -> SearchResult:
    return SearchResult(url=url, title=url, snippet="", relevance_score=0.9)


def test_citations_survive_the_re_research_loop():
    """
    increment_round() clears search_outputs, but approved claims from earlier
    rounds are kept. Building citations from search_outputs therefore dropped
    every round-1 source while the claims citing them stayed in the report.
    """
    from agents.graph import increment_round
    from agents.writer_agent import _build_citations

    state = ResearchState(query="q" * 12, max_rounds=2)

    # Round 1 finds a source.
    round_one = _source("https://round-one.example/a")
    state.search_outputs = [SearchAgentOutput(sub_question="sq1", results=[round_one])]
    state.all_sources[round_one.url] = round_one

    # The critic sends it round again; increment_round wipes search_outputs.
    import asyncio

    asyncio.run(increment_round(state))
    assert state.search_outputs == []

    # Round 2 finds a different source.
    round_two = _source("https://round-two.example/b")
    state.search_outputs = [SearchAgentOutput(sub_question="sq2", results=[round_two])]
    state.all_sources[round_two.url] = round_two

    cited = {c.url for c in _build_citations(state)}
    assert cited == {round_one.url, round_two.url}, (
        "round-1 sources must still be cited after the loop"
    )


# ---------------------------------------------------------------------------
# converged was hardcoded True
# ---------------------------------------------------------------------------
def _critic_result(needs_more: bool) -> CriticStructuredOutput:
    return CriticStructuredOutput(
        flagged_claims=[],
        overall_quality_score=0.5,
        coverage_score=0.5,
        source_diversity_score=0.5,
        contradiction_rate=0.0,
        needs_more_research=needs_more,
        critic_notes="",
    )


@pytest.mark.parametrize(
    ("needs_more", "current_round", "max_rounds", "expected_converged"),
    [
        (False, 0, 3, True),  # critic satisfied on round 1
        (True, 0, 3, True),  # wants more and can have it — not yet a failure
        (True, 2, 3, False),  # wants more, no rounds left → did NOT converge
        (False, 2, 3, True),  # satisfied on the last round
    ],
)
def test_converged_reflects_the_critic_not_a_constant(
    needs_more, current_round, max_rounds, expected_converged
):
    """
    The value the Writer reports as QualityReport.converged. It was set to True
    unconditionally in the Fact-Checker, which made the metric meaningless and
    the "did not converge" branch unreachable.
    """
    rounds_left = current_round < max_rounds - 1
    converged = not (needs_more and not rounds_left)
    assert converged is expected_converged


def test_fact_checker_does_not_force_convergence():
    """Guards the specific line that used to lie: state.converged = True."""
    import asyncio

    from agents.fact_checker_agent import run_fact_checking

    state = ResearchState(query="q" * 12)
    state.converged = False
    state.critic_output = CriticAgentOutput(
        flagged_claims=[],
        approved_claims=[],
        overall_quality_score=0.5,
        coverage_score=0.5,
        source_diversity_score=0.5,
        contradiction_rate=0.0,
        needs_more_research=False,
        critic_notes="",
    )

    asyncio.run(run_fact_checking(state))
    assert state.converged is False, "fact-checking must not manufacture convergence"


# ---------------------------------------------------------------------------
# Cost accounting dropped two agents
# ---------------------------------------------------------------------------
def test_analyst_and_fact_check_outputs_carry_cost():
    """
    Both agents unpacked `cost` from the LLM call and threw it away, so the
    per-report total omitted the two agents that fan out one call per
    sub-question / per flagged claim — i.e. most of the calls.
    """
    from agents.schemas import AnalystAgentOutput, ClaimVerdict, FactCheckResult

    assert "cost_usd" in AnalystAgentOutput.model_fields
    assert "cost_usd" in FactCheckResult.model_fields

    analyst = AnalystAgentOutput(
        sub_question="q",
        key_claims=[],
        contradictions_found=0,
        avg_confidence=0.0,
        cost_usd=0.25,
    )
    fc = FactCheckResult(
        claim_text="c",
        verdict=ClaimVerdict.VERIFIED,
        evidence="e",
        source_urls=[],
        confidence=ConfidenceLevel.HIGH,
        cost_usd=0.5,
    )
    assert analyst.cost_usd + fc.cost_usd == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# OpenAI parameter drift
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o", "max_tokens"),
        ("gpt-4o-mini-2024-07-18", "max_tokens"),
        ("o1", "max_completion_tokens"),
        ("o3-mini", "max_completion_tokens"),
        ("o4-mini-2025-04-16", "max_completion_tokens"),
        ("gpt-5", "max_completion_tokens"),
        ("gpt-5.1-mini", "max_completion_tokens"),
        ("llama-3.3-70b-versatile", "max_tokens"),
        ("gemini-1.5-flash", "max_tokens"),
    ],
)
def test_reasoning_models_get_max_completion_tokens(model, expected):
    assert _max_tokens_param_for(model) == expected


def test_cost_table_prefers_the_longest_prefix():
    """
    `gpt-4o-mini-2024-07-18` prefix-matches both `gpt-4o-mini` and `gpt-4o`,
    which is a 17x price difference. Correctness used to depend on dict
    insertion order.
    """
    assert compute_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    assert compute_cost("gpt-4o-2024-11-20", 1_000_000, 0) == pytest.approx(2.50)


# ---------------------------------------------------------------------------
# JSON extraction from chatty models
# ---------------------------------------------------------------------------
class _Tiny(BaseModel):
    a: int


@pytest.mark.parametrize(
    "reply",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the JSON you asked for:\n{"a": 1}',
        '{"a": 1}\n\nLet me know if you need anything else!',
        'Sure!\n```json\n{"a": 1}\n```\nThat covers the schema.',
    ],
)
def test_extract_json_survives_model_chatter(reply):
    assert _Tiny.model_validate(extract_json(reply)).a == 1


def test_extract_json_handles_a_nested_code_fence():
    """
    The old `text.split("```")[1]` took the *first* fenced block. A report whose
    body legitimately contains a fenced snippet lost everything after it.
    """
    reply = '{"a": 1, "note": "use ```py\\nprint()\\n``` for that"}'
    assert extract_json(reply)["a"] == 1


def test_extract_json_raises_when_there_is_no_json():
    with pytest.raises(ValueError, match="no JSON value"):
        extract_json("I could not complete that request.")

    with pytest.raises(ValueError, match="empty response"):
        extract_json("")


# ---------------------------------------------------------------------------
# Tavily relevance scores
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.8, 0.8), (None, 0.5), ("0.4", 0.4), (1.7, 1.0), (-2, 0.0), ("nope", 0.5)],
)
def test_relevance_score_is_always_in_range(raw, expected):
    """
    SearchResult constrains relevance_score to 0..1. `r.get("score", 0.5)` keeps
    a present-but-null value, and None fails validation — killing the whole
    search over one odd result.
    """
    score = _clamp_score(raw)
    assert score == pytest.approx(expected)
    SearchResult(url="u", title="t", snippet="s", relevance_score=score)


# ---------------------------------------------------------------------------
# Claim confidence — the triangulation rule the report is built on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("supporting", "contradicting", "expected"),
    [
        (3, 0, ConfidenceLevel.HIGH),
        (2, 0, ConfidenceLevel.MEDIUM),
        (1, 0, ConfidenceLevel.LOW),
        (0, 0, ConfidenceLevel.INCONCLUSIVE),
        (5, 1, ConfidenceLevel.CONTESTED),  # any contradiction wins
    ],
)
def test_confidence_assignment(supporting, contradicting, expected):
    from agents.analyst_agent import _assign_confidence

    assert _assign_confidence(supporting, contradicting) is expected


def test_claim_round_trips_through_state_serialisation():
    """
    LangGraph serialises state to a dict at every node boundary, so anything the
    graph carries has to survive model_dump() -> ResearchState(**d).
    """
    state = ResearchState(query="q" * 12)
    src = _source("https://example.com/x")
    state.all_sources[src.url] = src
    state.critic_output = CriticAgentOutput(
        flagged_claims=[],
        approved_claims=[
            Claim(
                text="t",
                source_urls=[src.url],
                confidence=ConfidenceLevel.HIGH,
                supporting_sources=3,
            )
        ],
        overall_quality_score=0.9,
        coverage_score=0.9,
        source_diversity_score=0.9,
        contradiction_rate=0.0,
        needs_more_research=False,
        critic_notes="",
    )

    restored = ResearchState(**state.model_dump())
    assert restored.all_sources[src.url].url == src.url
    assert restored.critic_output.approved_claims[0].text == "t"


# ---------------------------------------------------------------------------
# Writer output budget
# ---------------------------------------------------------------------------
def test_writer_cap_is_large_enough_for_a_whole_report():
    """
    The Writer's cap was 4096 tokens. A general report -- executive summary,
    4-6 key findings, 3-5 detailed sections, confidence assessment, limitations
    -- does not fit, and a live run hit the cap mid-sentence. Because the
    response is JSON, truncation loses the *entire* report rather than the tail
    of it, and it presents as "malformed JSON" rather than "ran out of room".

    The retry must get more room than the attempt that overflowed: a truncated
    document cannot be reproduced in the space that truncated it.
    """
    from agents.writer_agent import WRITER_MAX_TOKENS, WRITER_RETRY_MAX_TOKENS

    assert WRITER_MAX_TOKENS >= 12_000
    assert WRITER_RETRY_MAX_TOKENS > WRITER_MAX_TOKENS


def test_truncated_json_is_not_mistaken_for_malformed_json():
    """
    Truncation and malformation need opposite responses -- rewrite versus
    repair -- so extract_json must reject a cut-off document rather than
    returning some complete inner fragment it happens to contain.
    """
    truncated = (
        '{"title": "T", "key_findings": ["a", "b"], '
        '"detailed_sections": {"S": "text that stops mid-sen'
    )
    with pytest.raises(ValueError):
        extract_json(truncated, expect=dict)

    # The inner array is complete and would satisfy a naive scan.
    assert extract_json(truncated) == ["a", "b"]
