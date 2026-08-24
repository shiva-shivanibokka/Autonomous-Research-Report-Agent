"use client";

import { useEffect, useRef, useState } from "react";
import ControlPanel from "@/components/ControlPanel";
import Pipeline from "@/components/Pipeline";
import ActivityLog from "@/components/ActivityLog";
import QualityPanel from "@/components/QualityPanel";
import ReplayBanner from "@/components/ReplayBanner";
import ReportView from "@/components/ReportView";
import { useReplay } from "@/hooks/useReplay";
import { isReplayMode } from "@/lib/demo";
import {
  generateReport,
  getReport,
  getStatus,
  type GenerateArgs,
} from "@/lib/api";
import type { ReportResponse, StatusResponse } from "@/lib/types";

type Phase = "idle" | "running" | "done" | "error";

const POLL_MS = 2500;

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  // Replay mode is decided on the client: it depends on window.location, and
  // deciding it during SSR would hydrate the wrong UI. Undefined until mounted.
  const [replayMode, setReplayMode] = useState<boolean | null>(null);
  useEffect(() => setReplayMode(isReplayMode()), []);
  const replay = useReplay();

  function stopPolling() {
    if (timer.current) {
      clearInterval(timer.current);
      timer.current = null;
    }
  }

  useEffect(() => stopPolling, []);

  async function run(args: GenerateArgs) {
    stopPolling();
    setPhase("running");
    setStatus(null);
    setReport(null);
    setError(null);

    let jobId: string;
    try {
      ({ job_id: jobId } = await generateReport(args));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start the job.");
      setPhase("error");
      return;
    }

    timer.current = setInterval(async () => {
      try {
        const s = await getStatus(jobId);
        setStatus(s);
        if (s.status === "completed") {
          stopPolling();
          setReport(await getReport(jobId));
          setPhase("done");
        } else if (s.status === "failed") {
          stopPolling();
          setError(s.error || "The research job failed.");
          setPhase("error");
        }
      } catch (e) {
        // Transient poll error — keep trying; surface only if it persists.
        console.error(e);
      }
    }, POLL_MS);
  }

  function reset() {
    stopPolling();
    setPhase("idle");
    setStatus(null);
    setReport(null);
    setError(null);
    replay.reset();
  }

  async function playReplay() {
    setPhase("running");
    await replay.start();
  }

  async function skipReplay() {
    setPhase("running");
    await replay.skip();
  }

  const replaying = replayMode === true && (replay.playing || replay.finished);
  const activity = replaying ? replay.entries : (status?.activity_log ?? []);

  // A failed recording fetch leaves playing and finished both false, so
  // `replaying` is false while `phase` is still "running" — which rendered the
  // loading skeleton forever with nothing to explain it. Surface it instead: a
  // read that cannot be served may degrade, but it may not sit there implying
  // work is still happening.
  const shownError = replay.error ?? (phase === "error" ? error : null);

  return (
    <main className="mx-auto max-w-6xl px-5 py-10 sm:px-8 sm:py-14">
      <Header />

      {phase === "idle" ? (
        <section className="mt-12 grid gap-10 lg:grid-cols-[1.05fr_1fr] lg:items-start">
          <div>
            <p className="eyebrow">Multi-agent research pipeline</p>
            <h1 className="mt-4 font-display text-4xl font-semibold leading-[1.08] tracking-tight sm:text-5xl">
              From a question to a{" "}
              <span className="text-signal">cited, quality-scored</span> report —
              autonomously.
            </h1>
            <p className="mt-5 max-w-xl text-[15px] leading-7 text-text-muted">
              Seven specialized agents decompose your question, search and scrape
              the open web, triangulate sources, and run a self-improving critic
              loop before a writer synthesizes the findings. Every claim is
              traced to a source and scored for confidence.
            </p>
            <p className="mt-3 max-w-xl text-[15px] leading-7 text-text-muted">
              Bring your own key — Anthropic, OpenAI, Google, or Groq. Your key is
              sent per request and never stored.
            </p>
            <div className="mt-8">
              <Pipeline activity={[]} round={0} />
            </div>
          </div>
          {replayMode ? (
            <ReplayInvite onPlay={playReplay} onSkip={skipReplay} />
          ) : (
            <ControlPanel onSubmit={run} running={false} />
          )}
        </section>
      ) : (
        <section className="mt-10 space-y-4">
          {replaying && replay.run && (
            <ReplayBanner run={replay.run} playing={replay.playing} />
          )}
          <RunHeader
            status={status}
            report={report}
            phase={replay.error ? "error" : phase}
            onReset={reset}
            replayQuery={replaying ? replay.run?.query : undefined}
            replayTokens={replaying ? replay.tokens : undefined}
            replayCost={replaying ? replay.cost : undefined}
            replayDone={replay.finished}
            onSkip={replay.playing ? skipReplay : undefined}
          />
          <Pipeline
            activity={activity}
            round={replaying ? replay.round : (status?.current_round ?? 0)}
          />

          <div className="grid gap-4 lg:grid-cols-2">
            <ActivityLog entries={activity} />
            {replaying && replay.finished && replay.run ? (
              <QualityPanel quality={replay.run.quality} />
            ) : report ? (
              <QualityPanel quality={report.quality} />
            ) : (
              <RunningPlaceholder error={shownError} />
            )}
          </div>

          {replaying && replay.finished && replay.run ? (
            <ReportView markdown={replay.run.report_markdown} />
          ) : (
            report && <ReportView markdown={report.report_markdown} />
          )}
        </section>
      )}

      <footer className="mt-16 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-6 font-mono text-[11px] text-text-faint">
        <span>Autonomous Research Report Agent · LangGraph · FastAPI · BYOK</span>
        <span>
          Built by{" "}
          <a
            href="https://github.com/shiva-shivanibokka"
            target="_blank"
            rel="noreferrer"
            className="text-text-muted underline-offset-2 hover:text-text hover:underline"
          >
            Shivani Bokka
          </a>
        </span>
      </footer>
    </main>
  );
}

function Header() {
  return (
    <header className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <span className="grid h-9 w-9 place-items-center rounded-lg border border-signal/50 bg-signal/10 font-display text-sm font-bold text-signal">
          R
        </span>
        <div className="leading-none">
          <div className="font-mono text-[11px] uppercase tracking-[0.24em] text-text-faint">
            Research
          </div>
          <div className="font-display text-lg font-semibold tracking-tight">
            Agent
          </div>
        </div>
      </div>
      <span className="hidden font-mono text-[11px] uppercase tracking-wider text-text-muted sm:inline">
        BYOK · Anthropic / OpenAI / Google / Groq
      </span>
    </header>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-right">
      <div className="font-display text-lg font-semibold text-text">{value}</div>
      <div className="font-mono text-[10px] uppercase tracking-wider text-text-faint">
        {label}
      </div>
    </div>
  );
}

function RunHeader({
  status,
  report,
  phase,
  onReset,
  replayQuery,
  replayTokens,
  replayCost,
  replayDone,
  onSkip,
}: {
  status: StatusResponse | null;
  report: ReportResponse | null;
  phase: Phase;
  onReset: () => void;
  replayQuery?: string;
  replayTokens?: number;
  replayCost?: number;
  replayDone?: boolean;
  onSkip?: () => void;
}) {
  const isReplay = replayQuery !== undefined;
  const tokens = replayTokens ?? report?.tokens_used ?? status?.tokens_used ?? 0;
  const cost = replayCost ?? report?.cost_usd ?? status?.cost_usd ?? 0;
  const complete = phase === "done" || (isReplay && replayDone);
  const pill = complete
    ? { t: "complete", c: "border-verified/50 text-verified bg-verified/10" }
    : phase === "error"
      ? { t: "failed", c: "border-alert/50 text-alert bg-alert/10" }
      : { t: "running", c: "border-signal/50 text-signal bg-signal/10 animate-pulse-signal" };

  return (
    <div className="panel flex flex-wrap items-start justify-between gap-4 p-5">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span
            className={`rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${pill.c}`}
          >
            {pill.t}
          </span>
          <button
            onClick={onReset}
            className="font-mono text-[11px] uppercase tracking-wider text-text-faint underline-offset-2 hover:text-text hover:underline"
          >
            {isReplay ? "back" : "new research"}
          </button>
          {onSkip && (
            <button
              onClick={onSkip}
              className="font-mono text-[11px] uppercase tracking-wider text-text-faint underline-offset-2 hover:text-text hover:underline"
            >
              skip to result
            </button>
          )}
        </div>
        <p className="mt-2 truncate pr-4 text-[15px] text-text">
          {replayQuery ?? status?.query ?? report?.query}
        </p>
      </div>
      <div className="flex items-center gap-6">
        <Metric label="tokens" value={tokens.toLocaleString()} />
        <Metric label="est. cost" value={`$${cost.toFixed(4)}`} />
        {report && !isReplay && (
          <Metric label="duration" value={`${report.duration_seconds.toFixed(0)}s`} />
        )}
      </div>
    </div>
  );
}

function ReplayInvite({
  onPlay,
  onSkip,
}: {
  onPlay: () => void;
  onSkip: () => void;
}) {
  return (
    <div className="panel p-5 sm:p-6">
      <span className="eyebrow">Recorded run</span>
      <h2 className="mt-3 font-display text-xl font-semibold tracking-tight">
        Watch the pipeline work
      </h2>
      <p className="mt-3 text-[14.5px] leading-7 text-text-muted">
        This page has no backend to call — a report takes several minutes,
        launches headless Chromium and holds the scraped pages in memory, which
        no free host will run. So the pipeline was run for real once and
        recorded: the agent feed, the citations, the quality scores and the
        report below are all from that run.
      </p>
      <p className="mt-3 text-[14.5px] leading-7 text-text-muted">
        The question it was given is one where the evidence genuinely conflicts,
        which is the part worth watching — the critic loop and the contradiction
        map exist for exactly that case.
      </p>
      <div className="mt-6 flex flex-wrap gap-3">
        <button
          onClick={onPlay}
          className="rounded-lg border border-signal/60 bg-signal/15 px-6 py-3 font-display text-sm font-semibold uppercase tracking-[0.14em] text-signal shadow-glow transition hover:bg-signal/25"
        >
          Play the recorded run
        </button>
        <button
          onClick={onSkip}
          className="font-mono text-[12px] uppercase tracking-wider text-text-faint underline-offset-2 hover:text-text hover:underline"
        >
          skip to the report
        </button>
      </div>
      <p className="mt-6 border-t border-line pt-4 font-mono text-[11px] leading-5 text-text-faint">
        To run it live against your own key, see{" "}
        <a
          href="https://github.com/shiva-shivanibokka/Autonomous-Research-Report-Agent#run-it-locally"
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2 hover:text-text"
        >
          Run it locally
        </a>{" "}
        — two commands, no database.
      </p>
    </div>
  );
}

function RunningPlaceholder({ error }: { error: string | null }) {
  if (error) {
    return (
      <div className="panel p-5">
        <span className="eyebrow text-alert">Job failed</span>
        <p className="mt-3 font-mono text-[13px] leading-6 text-text-muted">
          {error}
        </p>
      </div>
    );
  }
  return (
    <div className="panel flex min-h-[16rem] flex-col justify-center p-5">
      <span className="eyebrow mb-3">Quality assessment</span>
      <div className="space-y-3">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-1.5 overflow-hidden rounded-full bg-ink-700">
            <div className="h-full w-1/3 animate-sweep bg-gradient-to-r from-transparent via-line to-transparent" />
          </div>
        ))}
      </div>
      <p className="mt-4 font-mono text-[11px] text-text-faint">
        Scores populate once the critic and fact-checker complete.
      </p>
    </div>
  );
}
