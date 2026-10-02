"""
End-to-end run of the compiled LangGraph pipeline with a scripted fake LLM,
fake search and fake scraper. No network, no keys.

The point is to exercise the paths that only run during a re-research round —
in particular that claims the Critic approved in round 1 still reach the
round-2 Critic and the Writer after increment_round() clears analyst_outputs.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import agents.fact_checker_agent as fact_checker_agent
import agents.llm_client as llm_client
import agents.scraper_agent as scraper_agent
import agents.search_agent as search_agent
import agents.writer_agent as writer_agent
from agents.graph import run_pipeline
from agents.schemas import ResearchState, ScrapedPage, SearchResult

ROUND1_GOOD = "Round-one claim that the critic approves."
ROUND1_WEAK = "Round-one weak claim that the critic flags."
ROUND2_NEW = "Round-two claim found by re-research."


class FakeLLM:
    """Routes on agent_name and records every prompt it was given."""

    def __init__(self, critic_wants_more_in_round1: bool = True):
        self.prompts: dict[str, list[str]] = {}
        self.critic_calls = 0
        self.analyst_calls = 0
        self.critic_wants_more_in_round1 = critic_wants_more_in_round1

    async def __call__(self, *, system, user, agent_name="unknown", **_):
        self.prompts.setdefault(agent_name, []).append(user)
        if agent_name == "orchestrator":
            body = {"sub_questions": ["sub question one"], "reasoning": "r"}
        elif agent_name == "analyst":
            self.analyst_calls += 1
            if self.critic_calls == 0:  # still round 1
                claims = [
                    {
                        "text": ROUND1_GOOD,
                        "source_urls": ["https://a.example/1"],
                        "supporting_sources": 3,
                        "contradicting_sources": 0,
                    },
                    {
                        "text": ROUND1_WEAK,
                        "source_urls": ["https://a.example/1"],
                        "supporting_sources": 1,
                        "contradicting_sources": 0,
                    },
                ]
            else:
                claims = [
                    {
                        "text": ROUND2_NEW,
                        "source_urls": ["https://b.example/2"],
                        "supporting_sources": 2,
                        "contradicting_sources": 0,
                    }
                ]
            body = {
                "claims": claims,
                "contradictions_found": 0,
                "avg_confidence_score": 0.5,
            }
        elif agent_name == "critic":
            self.critic_calls += 1
            first = self.critic_calls == 1
            body = {
                "flagged_claims": (
                    [
                        {
                            "claim_text": ROUND1_WEAK,
                            "reason": "single source",
                            "re_search_query": "verify weak claim",
                        }
                    ]
                    if first
                    else []
                ),
                "overall_quality_score": 0.5 if first else 0.8,
                "coverage_score": 0.6 if first else 0.9,
                "source_diversity_score": 0.5,
                "contradiction_rate": 0.0,
                "needs_more_research": first and self.critic_wants_more_in_round1,
                "critic_notes": "n",
            }
        elif agent_name == "fact_checker":
            body = {
                "verdict": "verified",
                "evidence": "e",
                "confidence": "medium",
                "explanation": "x",
            }
        elif agent_name == "writer":
            body = {
                "title": "T",
                "executive_summary": "S",
                "key_findings": ["k"],
                "detailed_sections": {"A": "a"},
                "confidence_assessment": "c",
                "limitations": "l",
            }
        else:  # pragma: no cover - unexpected agent
            raise AssertionError(f"unexpected agent {agent_name}")
        return json.dumps(body), 10, 10, 0.0


@pytest.fixture
def fake_env(monkeypatch):
    def install(fake: FakeLLM):
        # call_llm_structured looks call_llm up in llm_client's globals; the
        # Writer imported call_llm by name, so patch both bindings.
        monkeypatch.setattr(llm_client, "call_llm", fake)
        monkeypatch.setattr(writer_agent, "call_llm", fake)

        async def fake_search(query, max_results=8, **_):
            host = "a.example" if "one" in query else "b.example"
            return [
                SearchResult(
                    url=f"https://{host}/{i}",
                    title=f"t{i}",
                    snippet="s",
                    relevance_score=0.9,
                    source_domain=host,
                )
                for i in range(1, 3)
            ]

        async def fake_scrape(urls_with_titles, max_concurrent=3):
            return [
                ScrapedPage(url=u, title=t, content="page text", word_count=2)
                for u, t in urls_with_titles
            ]

        monkeypatch.setattr(search_agent, "tavily_search", fake_search)
        monkeypatch.setattr(fact_checker_agent, "tavily_search", fake_search)
        monkeypatch.setattr(scraper_agent, "scrape_pages", fake_scrape)

    return install


def _run(max_rounds: int) -> ResearchState:
    state = ResearchState(query="What is the test query about?", max_rounds=max_rounds)
    return asyncio.run(run_pipeline(state))


def test_round_one_claims_reach_the_writer_after_re_research(fake_env):
    fake = FakeLLM()
    fake_env(fake)
    final = _run(max_rounds=2)

    assert final.fatal_error is None
    assert final.current_round == 1, "the critic loop should have fired once"
    assert fake.critic_calls == 2

    # The round-2 Critic reviews the carried round-1 claim, not only new ones.
    assert ROUND1_GOOD in fake.prompts["critic"][1]
    assert ROUND2_NEW in fake.prompts["critic"][1]

    # The Writer sees both the approved round-1 claim and the round-2 claim...
    writer_prompt = fake.prompts["writer"][0]
    assert ROUND1_GOOD in writer_prompt
    assert ROUND2_NEW in writer_prompt
    # ...but not the round-1 claim the Critic flagged and sent for re-research.
    assert ROUND1_WEAK not in writer_prompt

    quality = final.final_report["quality"]
    assert quality["total_claims_extracted"] == 2
    assert quality["re_research_rounds"] == 2


def test_single_round_sends_flags_to_fact_checker(fake_env):
    """With max_rounds=1 the Critic's flags go to the Fact-Checker, not a loop."""
    fake = FakeLLM()
    fake_env(fake)
    final = _run(max_rounds=1)

    assert final.current_round == 0
    assert fake.critic_calls == 1
    assert len(final.fact_check_results) == 1
    assert final.fact_check_results[0].verdict.value == "verified"
    # It wanted another round and could not have one: not converged.
    assert final.converged is False
    assert "[FC-VERIFIED]" in fake.prompts["writer"][0]


def test_two_round_run_never_fact_checks_round_one_flags(fake_env):
    """
    Documents current behaviour: in a 2-round run the round-1 flags trigger
    re-research instead of fact-checking, and the Fact-Checker only sees flags
    from the final Critic call (none here), so it is skipped entirely.
    """
    fake = FakeLLM()
    fake_env(fake)
    final = _run(max_rounds=2)
    assert final.fact_check_results == []
    assert "fact_checker" not in fake.prompts
