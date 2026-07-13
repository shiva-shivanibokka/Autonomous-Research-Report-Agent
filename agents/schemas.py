"""
Pydantic schemas for every agent boundary in the research pipeline.
Every agent consumes and produces one of these models — no raw dicts passed between agents.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReportMode(str, Enum):
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    INVESTMENT_THESIS = "investment_thesis"
    ACADEMIC_LITERATURE_REVIEW = "academic_literature_review"
    GENERAL = "general"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"


class ConfidenceLevel(str, Enum):
    HIGH = "high"  # 3+ independent sources agree
    MEDIUM = "medium"  # 2 sources agree
    LOW = "low"  # single source
    CONTESTED = "contested"  # sources contradict each other
    INCONCLUSIVE = "inconclusive"  # insufficient evidence


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ClaimVerdict(str, Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


# ---------------------------------------------------------------------------
# Search layer
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """One result returned by the Search Agent via Tavily."""

    schema_version: str = "1.0"
    url: str
    title: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    published_date: str | None = None
    source_domain: str | None = None


class SearchAgentOutput(BaseModel):
    """Full output of one Search Agent run."""

    schema_version: str = "1.0"
    sub_question: str
    results: list[SearchResult]
    tokens_used: int = 0
    duration_seconds: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Scraper layer
# ---------------------------------------------------------------------------


class ScrapedPage(BaseModel):
    """One scraped and cleaned web page."""

    schema_version: str = "1.0"
    url: str
    title: str
    content: str  # cleaned text, no HTML
    word_count: int
    used_playwright: bool = False
    scrape_error: str | None = None


class ScraperAgentOutput(BaseModel):
    schema_version: str = "1.0"
    sub_question: str
    pages: list[ScrapedPage]
    tokens_used: int = 0
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Analyst layer
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """A single extracted claim from scraped content."""

    schema_version: str = "1.0"
    text: str
    source_urls: list[str]  # which pages this claim came from
    confidence: ConfidenceLevel
    supporting_sources: int  # number of independent sources that agree
    contradicting_sources: int = 0
    data_point: str | None = None  # any numeric/statistical value embedded in the claim
    contradiction_detail: str | None = None  # what the contradiction is, if any


class AnalystAgentOutput(BaseModel):
    schema_version: str = "1.0"
    sub_question: str
    key_claims: list[Claim]
    contradictions_found: int
    avg_confidence: float = Field(ge=0.0, le=1.0)
    tokens_used: int = 0
    duration_seconds: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Critic layer
# ---------------------------------------------------------------------------


class FlaggedClaim(BaseModel):
    """A claim the Critic has flagged for re-research."""

    claim_text: str
    reason: str  # why it was flagged
    sub_question: str  # which branch it belongs to
    re_search_query: str  # what to search for to verify it


class CriticAgentOutput(BaseModel):
    schema_version: str = "1.0"
    flagged_claims: list[FlaggedClaim]
    approved_claims: list[Claim]
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    source_diversity_score: float = Field(ge=0.0, le=1.0)
    contradiction_rate: float = Field(ge=0.0, le=1.0)
    needs_more_research: bool
    critic_notes: str
    tokens_used: int = 0
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Fact-checker layer
# ---------------------------------------------------------------------------


class FactCheckResult(BaseModel):
    schema_version: str = "1.0"
    claim_text: str
    verdict: ClaimVerdict
    evidence: str
    source_urls: list[str]
    confidence: ConfidenceLevel
    tokens_used: int = 0
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Writer layer — report output schemas per mode
# ---------------------------------------------------------------------------


class ContradictionEntry(BaseModel):
    """One contradiction detected across sources."""

    topic: str
    source_a: str
    source_a_claim: str
    source_b: str
    source_b_claim: str
    resolution: str  # how the Writer resolved it


class Citation(BaseModel):
    index: int
    url: str
    title: str
    domain: str
    accessed_date: str


class ConfidenceDistribution(BaseModel):
    high: int = 0
    medium: int = 0
    low: int = 0
    contested: int = 0
    inconclusive: int = 0


class QualityReport(BaseModel):
    """Machine-readable quality metadata attached to every report."""

    schema_version: str = "1.0"
    coverage_score: float
    source_diversity_score: float
    contradiction_rate: float
    overall_quality_score: float
    confidence_distribution: ConfidenceDistribution
    re_research_rounds: int
    total_sources_consulted: int
    total_claims_extracted: int
    claims_flagged_by_critic: int
    claims_verified_by_fact_checker: int
    converged: bool
    convergence_note: str


# -- General report (mode: general) --


class GeneralReport(BaseModel):
    schema_version: str = "1.0"
    title: str
    executive_summary: str
    key_findings: list[str]
    detailed_sections: dict[str, str]  # section_name -> content
    contradictions: list[ContradictionEntry]
    confidence_assessment: str
    limitations: str
    citations: list[Citation]
    quality: QualityReport


# -- Competitive Intelligence report --


class CompetitorEntry(BaseModel):
    name: str
    strengths: list[str]
    weaknesses: list[str]
    market_position: str
    notable_differentiators: list[str]


class CompetitiveIntelligenceReport(BaseModel):
    schema_version: str = "1.0"
    title: str
    executive_summary: str
    market_overview: str
    competitors: list[CompetitorEntry]
    competitive_matrix_summary: str
    whitespace_opportunities: list[str]
    strategic_recommendations: list[str]
    contradictions: list[ContradictionEntry]
    citations: list[Citation]
    quality: QualityReport


# -- Investment Thesis report --


class InvestmentThesisReport(BaseModel):
    schema_version: str = "1.0"
    title: str
    executive_summary: str
    market_sizing: dict[str, str]  # TAM, SAM, SOM with values
    thesis_statement: str
    bull_case: list[str]
    bear_case: list[str]
    risk_factors: list[str]
    comparable_transactions: list[str]
    key_metrics: dict[str, str]
    contradictions: list[ContradictionEntry]
    citations: list[Citation]
    quality: QualityReport


# -- Academic Literature Review report --


class AcademicLitReviewReport(BaseModel):
    schema_version: str = "1.0"
    title: str
    abstract: str
    introduction: str
    related_works: list[dict[str, str]]  # title, authors, year, summary, relevance
    methodology_comparison: str
    key_findings_synthesis: str
    research_gaps: list[str]
    future_directions: list[str]
    contradictions: list[ContradictionEntry]
    citations: list[Citation]
    quality: QualityReport


# Union type for all report modes
AnyReport = (
    GeneralReport
    | CompetitiveIntelligenceReport
    | InvestmentThesisReport
    | AcademicLitReviewReport
)


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------


class AgentActivityEntry(BaseModel):
    """One line in the live agent activity feed shown in the UI."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_name: str
    status: AgentStatus
    message: str
    tokens_used: int = 0
    cost_usd: float = 0.0


class ResearchState(BaseModel):
    """
    The single shared state object that flows through the entire LangGraph graph.
    Every node reads from and writes to this state.
    """

    # -- job identity --
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # -- input --
    query: str
    report_mode: ReportMode = ReportMode.GENERAL
    max_rounds: int = 3  # max critic/re-research iterations
    token_budget: int = 100_000  # total token budget for this job

    # -- BYOK LLM selection (api_key is in-memory only, never persisted to the DB) --
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-5"
    api_key: str | None = None  # None → server falls back to its own ANTHROPIC_API_KEY

    # -- orchestrator decomposition --
    sub_questions: list[str] = Field(default_factory=list)

    # -- search layer outputs --
    search_outputs: list[SearchAgentOutput] = Field(default_factory=list)

    # -- scraper layer outputs --
    scraper_outputs: list[ScraperAgentOutput] = Field(default_factory=list)

    # -- analyst layer outputs --
    analyst_outputs: list[AnalystAgentOutput] = Field(default_factory=list)

    # -- critic output --
    critic_output: CriticAgentOutput | None = None

    # -- fact-checker outputs (from dynamic spawning) --
    fact_check_results: list[FactCheckResult] = Field(default_factory=list)

    # -- final report --
    final_report: dict[str, Any] | None = None  # serialized AnyReport

    # -- self-improving loop tracking --
    current_round: int = 0
    converged: bool = False

    # -- token & cost accounting --
    tokens_used_total: int = 0
    cost_usd_total: float = 0.0
    tokens_by_agent: dict[str, int] = Field(default_factory=dict)

    # -- live activity feed --
    activity_log: list[AgentActivityEntry] = Field(default_factory=list)

    # -- error tracking --
    errors: list[str] = Field(default_factory=list)
    fatal_error: str | None = None

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# API request/response schemas
# ---------------------------------------------------------------------------


class ReportRequest(BaseModel):
    query: str = Field(min_length=10, max_length=500)
    report_mode: ReportMode = ReportMode.GENERAL
    max_rounds: int = Field(default=2, ge=1, le=3)
    token_budget: int = Field(default=80_000, ge=20_000, le=200_000)
    # BYOK: user picks a provider + model and supplies their own key (never stored).
    # Omit all three to use the server's own Anthropic key with a default model.
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-5"
    api_key: str | None = Field(default=None, max_length=400)


class ReportJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    query: str
    report_mode: str
    current_round: int
    tokens_used: int
    cost_usd: float
    activity_log: list[AgentActivityEntry]
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None


class ReportResponse(BaseModel):
    job_id: str
    query: str
    report_mode: str
    report: dict[str, Any]
    report_markdown: str = ""  # server-rendered Markdown for the UI
    quality: QualityReport
    tokens_used: int
    cost_usd: float
    duration_seconds: float
    created_at: datetime
    completed_at: datetime
