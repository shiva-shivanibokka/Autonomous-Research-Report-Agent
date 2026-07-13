import type {
  Provider,
  ReportMode,
  ReportResponse,
  StatusResponse,
} from "./types";

// Backend base URL. Set NEXT_PUBLIC_API_URL to your Cloud Run URL in production;
// defaults to the local FastAPI server for development.
const API_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function listModels(
  provider: Provider,
  apiKey: string,
): Promise<string[]> {
  const res = await fetch(`${API_URL}/providers/${provider}/models`, {
    headers: { "X-Provider-Key": apiKey },
  });
  const data = await asJson<{ models: string[] }>(res);
  return data.models;
}

export interface GenerateArgs {
  query: string;
  reportMode: ReportMode;
  maxRounds: number;
  tokenBudget: number;
  provider: Provider;
  model: string;
  apiKey?: string;
}

export async function generateReport(
  args: GenerateArgs,
): Promise<{ job_id: string }> {
  const res = await fetch(`${API_URL}/report/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: args.query,
      report_mode: args.reportMode,
      max_rounds: args.maxRounds,
      token_budget: args.tokenBudget,
      provider: args.provider,
      model: args.model,
      api_key: args.apiKey || null,
    }),
  });
  return asJson<{ job_id: string }>(res);
}

export async function getStatus(jobId: string): Promise<StatusResponse> {
  const res = await fetch(`${API_URL}/report/status/${jobId}`);
  return asJson<StatusResponse>(res);
}

export async function getReport(jobId: string): Promise<ReportResponse> {
  const res = await fetch(`${API_URL}/report/${jobId}`);
  return asJson<ReportResponse>(res);
}
