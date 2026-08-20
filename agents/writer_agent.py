"""
Writer Agent — synthesizes all validated findings into a structured report.

Produces domain-specific output based on report_mode:
- GeneralReport
- CompetitiveIntelligenceReport
- InvestmentThesisReport
- AcademicLitReviewReport

Every report includes a QualityReport with machine-readable quality metrics
and a Contradiction Map section listing every detected conflict.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from urllib.parse import urlparse

import structlog
from pydantic import ValidationError

from agents.llm_client import REASON_MODEL, call_llm, extract_json
from agents.schemas import (
    AcademicLitReviewReport,
    AgentActivityEntry,
    AgentStatus,
    Citation,
    Claim,
    ClaimVerdict,
    CompetitiveIntelligenceReport,
    ConfidenceDistribution,
    ConfidenceLevel,
    ContradictionEntry,
    GeneralReport,
    InvestmentThesisReport,
    QualityReport,
    ReportMode,
    ResearchState,
)

log = structlog.get_logger(__name__)

# Max chars of claim context passed to the Writer LLM
MAX_WRITER_CONTEXT = 30_000

# Output cap for the report call and its repair retry.
WRITER_MAX_TOKENS = 4096

# Which schema each mode's report should satisfy. schemas.py promises "no raw
# dicts passed between agents"; the Writer was the one boundary still handing
# back unvalidated model output, so the report was the least checked object in
# the pipeline despite being the only one anybody reads.
REPORT_SCHEMA_BY_MODE = {
    ReportMode.GENERAL: GeneralReport,
    ReportMode.COMPETITIVE_INTELLIGENCE: CompetitiveIntelligenceReport,
    ReportMode.INVESTMENT_THESIS: InvestmentThesisReport,
    ReportMode.ACADEMIC_LITERATURE_REVIEW: AcademicLitReviewReport,
}


def _build_citations(state: ResearchState) -> list[Citation]:
    """Collect all unique URLs across every research round and number them."""
    seen: dict[str, Citation] = {}
    idx = 1
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    # state.all_sources spans every round; state.search_outputs is only the last
    # one. Reading the accumulator is what keeps round-1 claims cited after the
    # critic loop fires.
    for result in state.all_sources.values():
        if result.url not in seen:
            seen[result.url] = Citation(
                index=idx,
                url=result.url,
                title=result.title[:120],
                domain=result.source_domain or urlparse(result.url).netloc,
                accessed_date=today,
            )
            idx += 1

    # Also add fact-check sources
    for fc_result in state.fact_check_results:
        for url in fc_result.source_urls:
            if url not in seen:
                domain = urlparse(url).netloc
                seen[url] = Citation(
                    index=idx,
                    url=url,
                    title=domain,
                    domain=domain,
                    accessed_date=today,
                )
                idx += 1

    return list(seen.values())


def _build_quality_report(
    state: ResearchState, citations: list[Citation]
) -> QualityReport:
    """Build machine-readable quality metrics for the report."""
    all_claims: list[Claim] = []
    for a in state.analyst_outputs:
        all_claims.extend(a.key_claims)

    dist = ConfidenceDistribution(
        high=sum(1 for c in all_claims if c.confidence == ConfidenceLevel.HIGH),
        medium=sum(1 for c in all_claims if c.confidence == ConfidenceLevel.MEDIUM),
        low=sum(1 for c in all_claims if c.confidence == ConfidenceLevel.LOW),
        contested=sum(
            1 for c in all_claims if c.confidence == ConfidenceLevel.CONTESTED
        ),
        inconclusive=sum(
            1 for c in all_claims if c.confidence == ConfidenceLevel.INCONCLUSIVE
        ),
    )

    critic = state.critic_output
    verified_count = sum(
        1 for r in state.fact_check_results if r.verdict == ClaimVerdict.VERIFIED
    )

    converged = state.converged
    rounds = state.current_round + 1
    convergence_note = (
        f"Report converged after {rounds} research round(s)."
        if converged
        else (
            f"Did not converge: {rounds} of {state.max_rounds} round(s) used and "
            "the Critic's quality bar was not met."
        )
    )
    if state.fact_check_results:
        inconclusive = sum(
            1
            for r in state.fact_check_results
            if r.verdict == ClaimVerdict.INCONCLUSIVE
        )
        if inconclusive:
            convergence_note += f" {inconclusive} claim(s) remain inconclusive."

    return QualityReport(
        coverage_score=critic.coverage_score if critic else 0.5,
        source_diversity_score=critic.source_diversity_score if critic else 0.5,
        contradiction_rate=critic.contradiction_rate if critic else 0.0,
        overall_quality_score=critic.overall_quality_score if critic else 0.5,
        confidence_distribution=dist,
        re_research_rounds=rounds,
        total_sources_consulted=len(citations),
        total_claims_extracted=len(all_claims),
        claims_flagged_by_critic=len(critic.flagged_claims) if critic else 0,
        claims_verified_by_fact_checker=verified_count,
        converged=converged,
        convergence_note=convergence_note,
    )


def _build_contradiction_map(state: ResearchState) -> list[ContradictionEntry]:
    """Extract detected contradictions and fact-check results into a structured map."""
    entries: list[ContradictionEntry] = []

    # From Analyst-detected contradictions
    for analyst_out in state.analyst_outputs:
        for claim in analyst_out.key_claims:
            if claim.contradiction_detail and claim.contradicting_sources > 0:
                sources = claim.source_urls
                source_a = sources[0] if sources else "Source A"
                source_b = sources[1] if len(sources) > 1 else "Source B"
                entries.append(
                    ContradictionEntry(
                        topic=claim.text[:80],
                        source_a=source_a,
                        source_a_claim=claim.text,
                        source_b=source_b,
                        source_b_claim=claim.contradiction_detail,
                        resolution="Claim retained with CONTESTED confidence — conflicting evidence noted.",
                    )
                )

    # From Fact-Checker contradictions
    for fc in state.fact_check_results:
        if fc.verdict == ClaimVerdict.CONTRADICTED:
            entries.append(
                ContradictionEntry(
                    topic=fc.claim_text[:80],
                    source_a="Original source",
                    source_a_claim=fc.claim_text,
                    source_b=fc.source_urls[0]
                    if fc.source_urls
                    else "Verification source",
                    source_b_claim=fc.evidence[:200],
                    resolution="Claim marked as CONTRADICTED by independent fact-check.",
                )
            )

    return entries[:10]  # cap at 10 entries for readability


def _build_claims_context(state: ResearchState) -> str:
    """Build a compact summary of all approved claims for the Writer LLM."""
    lines = []
    if state.critic_output:
        claims = state.critic_output.approved_claims
    else:
        claims = [c for a in state.analyst_outputs for c in a.key_claims]

    for i, claim in enumerate(claims, 1):
        conf = claim.confidence.value.upper()
        lines.append(f"[{i}] [{conf}] {claim.text}")
        if claim.data_point:
            lines.append(f"    Data: {claim.data_point}")
        if claim.source_urls:
            lines.append(f"    Sources: {', '.join(claim.source_urls[:2])}")

    # Also include verified fact-checks
    for fc in state.fact_check_results:
        if fc.verdict == ClaimVerdict.VERIFIED:
            lines.append(f"[FC-VERIFIED] {fc.claim_text}")

    return "\n".join(lines)[:MAX_WRITER_CONTEXT]


async def run_writer_agent(state: ResearchState) -> ResearchState:
    """
    LangGraph node: Synthesize all validated findings into the final report.
    """
    log.info("writer_agent_start", job_id=state.job_id, mode=state.report_mode.value)

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Writer Agent",
            status=AgentStatus.RUNNING,
            message=f"Writing {state.report_mode.value} report...",
        )
    )

    citations = _build_citations(state)
    quality = _build_quality_report(state, citations)
    contradictions = _build_contradiction_map(state)
    claims_context = _build_claims_context(state)

    # Build citation reference string for the Writer
    citation_refs = "\n".join(
        f"[{c.index}] {c.title} — {c.url}" for c in citations[:30]
    )

    mode = state.report_mode
    t0 = time.perf_counter()

    # ---- Mode-specific prompt and structure ----
    if mode == ReportMode.COMPETITIVE_INTELLIGENCE:
        system = """You are a senior competitive intelligence analyst writing a professional report.
Structure your response as valid JSON with these exact fields:
{
  "title": "string",
  "executive_summary": "string (2-3 paragraphs)",
  "market_overview": "string (1-2 paragraphs)",
  "competitors": [
    {"name": "string", "strengths": ["..."], "weaknesses": ["..."], 
     "market_position": "string", "notable_differentiators": ["..."]}
  ],
  "competitive_matrix_summary": "string",
  "whitespace_opportunities": ["string"],
  "strategic_recommendations": ["string"]
}
Use the approved claims and cite sources by URL where possible."""

    elif mode == ReportMode.INVESTMENT_THESIS:
        system = """You are a senior investment analyst writing a thesis report.
Structure your response as valid JSON with these exact fields:
{
  "title": "string",
  "executive_summary": "string",
  "market_sizing": {"TAM": "string", "SAM": "string", "SOM": "string"},
  "thesis_statement": "string",
  "bull_case": ["string"],
  "bear_case": ["string"],
  "risk_factors": ["string"],
  "comparable_transactions": ["string"],
  "key_metrics": {"metric_name": "value"}
}"""

    elif mode == ReportMode.ACADEMIC_LITERATURE_REVIEW:
        system = """You are an academic researcher writing a literature review.
Structure your response as valid JSON with these exact fields:
{
  "title": "string",
  "abstract": "string",
  "introduction": "string",
  "related_works": [{"title": "string", "authors": "string", "year": "string", "summary": "string", "relevance": "string"}],
  "methodology_comparison": "string",
  "key_findings_synthesis": "string",
  "research_gaps": ["string"],
  "future_directions": ["string"]
}"""

    else:  # GENERAL
        system = """You are a research analyst writing a comprehensive report.
Structure your response as valid JSON with these exact fields:
{
  "title": "string",
  "executive_summary": "string (2-3 paragraphs)",
  "key_findings": ["string"],
  "detailed_sections": {"section_name": "section_content"},
  "confidence_assessment": "string",
  "limitations": "string"
}
Include 4-6 key findings and 3-5 detailed sections."""

    user = f"""Research query: {state.query}

Approved research claims ({quality.total_claims_extracted} total, {quality.claims_flagged_by_critic} flagged):
{claims_context}

Available citations:
{citation_refs}

Quality context:
- Coverage score: {quality.coverage_score:.2f}
- Source diversity: {quality.source_diversity_score:.2f}
- Research rounds: {quality.re_research_rounds}
- Convergence: {quality.convergence_note}

Write a comprehensive, well-structured report based on this research. 
Cite sources using their URLs. Be factual and precise."""

    try:
        text, inp, out, cost = await call_llm(
            system=system,
            user=user,
            model=REASON_MODEL,
            max_tokens=WRITER_MAX_TOKENS,
            temperature=0.3,
            agent_name="writer",
            token_budget_remaining=state.token_budget - state.tokens_used_total,
        )
    except Exception as e:
        log.error("writer_llm_failed", error=str(e))
        state.errors.append(f"Writer failed: {e}")
        # The Writer is the last node: with no report there is nothing to serve,
        # so this has to be fatal. Returning without it marked the job "completed"
        # with report=None, and GET /report/{id} then 404'd on a job the UI had
        # just shown as successful.
        state.fatal_error = f"Writer failed: {e}"
        state.activity_log.append(
            AgentActivityEntry(
                agent_name="Writer Agent",
                status=AgentStatus.FAILED,
                message=f"Writer failed: {str(e)[:200]}",
            )
        )
        return state

    duration = time.perf_counter() - t0

    # Parse writer output into a structured report. extract_json ignores fences
    # and any note the model appends after the closing brace.
    try:
        report_dict = extract_json(text, expect=dict)
    except ValueError as first_error:
        # Observed live: the model closed an object with `]`, making the whole
        # response unparseable. By this point the run has spent minutes and
        # thousands of tokens gathering the research, so one cheap repair call
        # is obviously worth it before falling back to raw text.
        log.warning("writer_json_invalid_attempting_repair", error=str(first_error))
        try:
            repaired, r_inp, r_out, r_cost = await call_llm(
                system=(
                    "You fix malformed JSON. Return ONLY the corrected JSON object. "
                    "Preserve all content exactly; change only the syntax. "
                    "Do not add commentary, explanation, or markdown fences."
                ),
                user=f"This JSON does not parse ({first_error}). Return it corrected:\n\n{text}",
                model=REASON_MODEL,
                max_tokens=WRITER_MAX_TOKENS,
                agent_name="writer_repair",
                token_budget_remaining=state.token_budget - state.tokens_used_total,
            )
            inp += r_inp
            out += r_out
            cost += r_cost
            report_dict = extract_json(repaired, expect=dict)
            log.info("writer_json_repaired", job_id=state.job_id)
        except Exception as repair_error:
            log.warning("writer_json_repair_failed", error=str(repair_error))
            report_dict = None

    if not isinstance(report_dict, dict):
        # Last resort: the research is done and paid for, so surface it as prose
        # rather than throwing away a completed pipeline over a formatting slip.
        # Strip the code fence so the reader gets text, not a wall of raw JSON.
        log.error("writer_json_unrecoverable", job_id=state.job_id)
        state.errors.append("Writer returned malformed JSON; raw output preserved.")
        body = text.strip()
        if body.startswith("```"):
            body = body.split("\n", 1)[-1]
        body = body.removesuffix("```").strip()
        report_dict = {
            "title": f"Research Report: {state.query[:80]}",
            "executive_summary": (
                "The report could not be parsed into sections. The model's full "
                "output is reproduced below."
            ),
            "key_findings": [],
            "detailed_sections": {"Full Report": body},
            "confidence_assessment": quality.convergence_note,
            "limitations": "Report parsing failed — raw model output included above.",
        }

    # Inject shared fields into all report modes
    report_dict["citations"] = [c.model_dump() for c in citations[:30]]
    report_dict["quality"] = quality.model_dump()
    report_dict["contradictions"] = [c.model_dump() for c in contradictions]
    report_dict.setdefault("title", f"Research Report: {state.query[:80]}")
    report_dict["schema_version"] = "1.0"

    # Validate against the mode's schema so a missing section is a logged,
    # visible defect rather than a key the frontend silently renders as blank.
    # Non-fatal on purpose: a report that is 90% there still beats no report,
    # and the reader is told what was missing.
    schema = REPORT_SCHEMA_BY_MODE.get(mode)
    if schema is not None:
        try:
            schema.model_validate(report_dict)
        except ValidationError as ve:
            missing = sorted({str(err["loc"][0]) for err in ve.errors() if err["loc"]})
            log.warning(
                "writer_report_schema_mismatch",
                job_id=state.job_id,
                mode=mode.value,
                fields=missing,
            )
            state.errors.append(
                f"Report did not match the {mode.value} schema: {', '.join(missing)}"
            )
            report_dict["limitations"] = (
                str(report_dict.get("limitations", "")).rstrip()
                + f"\n\nIncomplete sections (model omitted them): {', '.join(missing)}."
            ).strip()

    state.final_report = report_dict
    state.tokens_used_total += inp + out
    state.cost_usd_total += cost
    state.tokens_by_agent["writer"] = state.tokens_by_agent.get("writer", 0) + inp + out

    state.activity_log.append(
        AgentActivityEntry(
            agent_name="Writer Agent",
            status=AgentStatus.COMPLETED,
            message=f"Report written: '{report_dict.get('title', 'Untitled')}' | {len(citations)} citations | {len(contradictions)} contradictions mapped",
            tokens_used=inp + out,
            cost_usd=cost,
        )
    )

    log.info(
        "writer_agent_complete",
        job_id=state.job_id,
        mode=mode.value,
        citations=len(citations),
        contradictions=len(contradictions),
        tokens=inp + out,
        cost=cost,
        duration=round(duration, 3),
    )

    return state
