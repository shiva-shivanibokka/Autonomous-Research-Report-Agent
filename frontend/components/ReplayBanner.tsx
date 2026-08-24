"use client";

import { formatRecordedAt, type DemoRun } from "@/lib/demo";

/**
 * States plainly that this is a recording, and of what.
 *
 * A replay that does not say it is a replay is a lie told with real data. The
 * point of this banner is that a visitor can tell at a glance what they are
 * looking at, when it was captured, and what the run actually cost in wall-clock
 * time — which is the one thing playback compresses.
 */
export default function ReplayBanner({
  run,
  playing,
}: {
  run: DemoRun;
  playing: boolean;
}) {
  const recorded = formatRecordedAt(run.recorded_at);
  const mins = Math.floor(run.duration_seconds / 60);
  const secs = Math.round(run.duration_seconds % 60);
  const realTime = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;

  return (
    <div className="panel border-signal/40 bg-signal/[0.04] p-4">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="rounded-full border border-signal/50 bg-signal/10 px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-signal">
          {playing ? "replaying" : "recording"}
        </span>
        <span className="text-[15px] text-text">
          A real run of this pipeline, recorded {recorded && `on ${recorded}`}.
        </span>
      </div>
      <p className="mt-2 max-w-3xl text-[13.5px] leading-6 text-text-muted">
        Every agent message, token count, citation and dollar figure below is
        what actually happened — {run.sources_consulted} sources retrieved from
        live web search across {run.rounds_run} research{" "}
        {run.rounds_run === 1 ? "round" : "rounds"}, the highest-ranked of them
        scraped and read, {run.model} writing the report. The real run took{" "}
        <span className="text-text">{realTime}</span> and cost{" "}
        <span className="text-text">${run.cost_usd.toFixed(4)}</span>; playback is
        compressed to about half a minute, keeping the relative pacing of each
        stage.
      </p>
      <p className="mt-2 max-w-3xl text-[13px] leading-6 text-text-faint">
        There is no hosted backend: a report needs several minutes, headless
        Chromium and the scraped pages held in memory, which no free tier
        carries. Rather than leave a dead link, the run was recorded.{" "}
        <a
          href="https://github.com/shiva-shivanibokka/Autonomous-Research-Report-Agent#run-it-locally"
          target="_blank"
          rel="noreferrer"
          className="text-text-muted underline underline-offset-2 hover:text-text"
        >
          Run it yourself in two commands
        </a>{" "}
        with your own key.
      </p>
    </div>
  );
}
