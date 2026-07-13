export type Provider = "anthropic" | "openai" | "google" | "groq";

export type ReportMode =
  | "general"
  | "competitive_intelligence"
  | "investment_thesis"
  | "academic_literature_review";

export type JobStatus = "queued" | "running" | "completed" | "failed";

export type AgentStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "skipped";

export interface ActivityEntry {
  timestamp: string;
  agent_name: string;
  status: AgentStatus;
  message: string;
  tokens_used: number;
  cost_usd: number;
}

export interface StatusResponse {
  job_id: string;
  status: JobStatus;
  query: string;
  report_mode: ReportMode;
  current_round: number;
  tokens_used: number;
  cost_usd: number;
  activity_log: ActivityEntry[];
  created_at: string;
  completed_at: string | null;
  error: string | null;
}

export interface Quality {
  coverage_score: number;
  source_diversity_score: number;
  contradiction_rate: number;
  overall_quality_score: number;
  confidence_distribution: Record<string, number>;
  re_research_rounds: number;
  total_sources_consulted: number;
  total_claims_extracted: number;
  claims_flagged_by_critic: number;
  claims_verified_by_fact_checker: number;
  converged: boolean;
  convergence_note: string;
}

export interface ReportResponse {
  job_id: string;
  query: string;
  report_mode: ReportMode;
  report: Record<string, unknown> & { quality?: Quality };
  report_markdown: string;
  quality: Quality;
  tokens_used: number;
  cost_usd: number;
  duration_seconds: number;
}
