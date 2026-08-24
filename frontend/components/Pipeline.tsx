"use client";

import InfoTip from "@/components/InfoTip";
import type { ActivityEntry } from "@/lib/types";

// The pipeline IS the product — render it as an instrument panel where each
// agent stage illuminates as it runs. Stage state is derived from the live feed.
const STAGES = [
  { key: "orchestrate", label: "Orchestrate", match: ["orchestr"] },
  { key: "search", label: "Search", match: ["search"] },
  { key: "scrape", label: "Scrape", match: ["scrap"] },
  { key: "analyze", label: "Analyze", match: ["analy"] },
  { key: "critique", label: "Critique", match: ["critic"] },
  { key: "factcheck", label: "Fact-check", match: ["fact"] },
  { key: "write", label: "Write", match: ["writ"] },
] as const;

type StageState = "pending" | "running" | "completed" | "failed";

function stageState(entries: ActivityEntry[], match: readonly string[]): StageState {
  const mine = entries.filter((e) =>
    match.some((m) => e.agent_name.toLowerCase().includes(m)),
  );
  if (mine.length === 0) return "pending";
  const last = mine[mine.length - 1].status;
  if (last === "failed") return "failed";
  if (last === "completed" || last === "skipped") return "completed";
  return "running";
}

const DOT: Record<StageState, string> = {
  pending: "border-line bg-ink-700 text-text-faint",
  running: "border-signal bg-signal/15 text-signal shadow-glow animate-pulse-signal",
  completed: "border-verified bg-verified/10 text-verified",
  failed: "border-alert bg-alert/10 text-alert",
};

export default function Pipeline({
  activity,
  round,
}: {
  activity: ActivityEntry[];
  round: number;
}) {
  const states = STAGES.map((s) => stageState(activity, s.match));

  return (
    <div className="panel p-6">
      <div className="mb-4 flex items-center justify-between">
        <span className="flex items-center">
          <span className="eyebrow">Agent pipeline</span>
          <InfoTip text="The seven agents, in the order the graph runs them. A stage lights amber while it is working and turns cyan when it finishes. Critique can send the run back to Orchestrate for another round, which is the self-improving loop; Fact-check is skipped entirely when the Critic flags nothing." />
        </span>
        {round > 0 && (
          <span className="font-mono text-[12.5px] text-text-muted">
            re-research round {round + 1}
          </span>
        )}
      </div>

      <ol className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-0">
        {STAGES.map((stage, i) => {
          const state = states[i];
          const connectorLit =
            states[i] === "completed" &&
            (i + 1 >= STAGES.length || states[i + 1] !== "pending");
          return (
            <li
              key={stage.key}
              className="flex items-center gap-3 sm:flex-1 sm:flex-col sm:items-stretch sm:gap-0"
            >
              <div className="flex items-center sm:w-full">
                <span
                  className={`grid h-9 w-9 shrink-0 place-items-center rounded-lg border font-mono text-xs transition ${DOT[state]}`}
                  aria-label={`${stage.label}: ${state}`}
                >
                  {String(i + 1).padStart(2, "0")}
                </span>
                {i < STAGES.length - 1 && (
                  <div className="relative mx-2 hidden h-px flex-1 overflow-hidden bg-line sm:block">
                    <div
                      className={`absolute inset-0 ${
                        connectorLit ? "bg-verified/50" : ""
                      }`}
                    />
                    {states[i + 1] === "running" && (
                      <div className="absolute inset-y-0 w-1/2 animate-sweep bg-gradient-to-r from-transparent via-signal to-transparent" />
                    )}
                  </div>
                )}
              </div>
              <span
                className={`font-mono text-[12px] uppercase tracking-wider sm:mt-2 ${
                  state === "pending" ? "text-text-faint" : "text-text-muted"
                }`}
              >
                {stage.label}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
