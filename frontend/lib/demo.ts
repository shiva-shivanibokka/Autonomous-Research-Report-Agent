import type { ActivityEntry, Quality, ReportMode } from "./types";

/**
 * Replay of one real pipeline run, recorded by scripts/record_demo_run.py.
 *
 * This project cannot be hosted the ordinary way: a report takes minutes,
 * launches headless Chromium, and holds scraped pages in memory. Rather than
 * publish a dead link or fake the product, the real thing was run once and its
 * output committed. Everything below is what actually happened — the agent
 * messages, the token counts, the dollar figures, the report.
 */

export interface DemoFrame {
  /** Seconds into the real run when this entry appeared. */
  at: number;
  entry: ActivityEntry;
  round: number;
  tokens_used: number;
  cost_usd: number;
}

export interface DemoRun {
  schema_version: string;
  recorded_at: string;
  query: string;
  report_mode: ReportMode;
  provider: string;
  model: string;
  duration_seconds: number;
  tokens_used: number;
  cost_usd: number;
  tokens_by_agent: Record<string, number>;
  rounds_run: number;
  max_rounds: number;
  converged: boolean;
  sources_consulted: number;
  frames: DemoFrame[];
  activity_log: ActivityEntry[];
  report: Record<string, unknown>;
  report_markdown: string;
  quality: Quality;
}

/**
 * How long the replay takes to play back.
 *
 * The real run is minutes long, most of it spent waiting on the network. Nobody
 * watches a portfolio demo for that, so playback is compressed onto this
 * timeline while keeping the *relative* pacing — a step that really took four
 * times longer than another still looks four times longer. The banner states
 * the true duration, so the compression is visible rather than implied.
 */
export const REPLAY_SECONDS = 32;

const DEMO_MODE_ENV =
  process.env.NEXT_PUBLIC_DEMO_MODE === "1" ||
  process.env.NEXT_PUBLIC_DEMO_MODE === "true";

const API_CONFIGURED = Boolean(process.env.NEXT_PUBLIC_API_URL);

const LOCAL_HOSTS = /^(localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0)$/;

/**
 * Should the page offer the recording instead of a live backend?
 *
 * Yes when explicitly asked for, and also whenever the page is served from
 * somewhere that is not this machine with no backend URL configured — because
 * then there is provably nothing to talk to. `API_URL` falls back to
 * http://localhost:8000, which is a real address on a developer's laptop and a
 * dead one on a hosted deploy.
 *
 * Inferring it means a deploy cannot be quietly wrong: forgetting the
 * environment variable would otherwise leave a console hammering localhost
 * until it gave up, instead of showing the demo it was built for.
 */
export function isReplayMode(): boolean {
  if (DEMO_MODE_ENV) return true;
  if (API_CONFIGURED) return false;
  if (typeof window === "undefined") return false;
  return !LOCAL_HOSTS.test(window.location.hostname);
}

let cached: Promise<DemoRun> | null = null;

/** Load the recording. Cached so replaying twice does not refetch it. */
export function loadDemoRun(): Promise<DemoRun> {
  if (!cached) {
    cached = fetch("/demo/run.json").then((r) => {
      if (!r.ok) throw new Error(`Could not load the recording (${r.status}).`);
      return r.json() as Promise<DemoRun>;
    });
  }
  return cached;
}

/**
 * Map a frame's real timestamp onto the compressed playback timeline.
 *
 * Scaling by the last frame rather than by duration_seconds keeps the final
 * entry from landing early when the run spent its tail inside one long call.
 */
export function playbackSchedule(frames: DemoFrame[]): number[] {
  if (frames.length === 0) return [];
  const span = frames[frames.length - 1].at || 1;
  return frames.map((f) => (f.at / span) * REPLAY_SECONDS * 1000);
}

export function formatRecordedAt(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}
