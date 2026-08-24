"use client";

import { useEffect, useRef } from "react";
import InfoTip from "@/components/InfoTip";
import type { ActivityEntry, AgentStatus } from "@/lib/types";

const ICON: Record<AgentStatus, string> = {
  pending: "·",
  running: "▸",
  completed: "✓",
  failed: "✗",
  skipped: "—",
};

const TONE: Record<AgentStatus, string> = {
  pending: "text-text-faint",
  running: "text-signal",
  completed: "text-verified",
  failed: "text-alert",
  skipped: "text-text-faint",
};

export default function ActivityLog({ entries }: { entries: ActivityEntry[] }) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries.length]);

  return (
    <div className="panel flex h-full min-h-[16rem] flex-col p-6">
      <span className="mb-3 flex items-center">
        <span className="eyebrow">Telemetry</span>
        <InfoTip text="The live feed each agent writes as it works, with the tokens and cost it spent. This is the pipeline reporting on itself — the same entries are persisted to the job record, so a run that fails still says which agent broke and why." />
      </span>
      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1 font-mono text-[14px] leading-relaxed">
        {entries.length === 0 ? (
          <p className="text-text-faint">Awaiting first agent…</p>
        ) : (
          entries.map((e, i) => {
            const ts = e.timestamp?.slice(11, 19) ?? "";
            return (
              <div key={i} className="flex animate-rise gap-2.5">
                <span className="shrink-0 text-text-faint">{ts}</span>
                <span className={`shrink-0 ${TONE[e.status]}`}>
                  {ICON[e.status]}
                </span>
                <span className="min-w-0">
                  <span className="text-text">{e.agent_name}</span>
                  <span className="text-text-muted"> — {e.message}</span>
                  {e.tokens_used > 0 && (
                    <span className="text-text-faint">
                      {"  "}·{"  "}
                      {e.tokens_used.toLocaleString()} tok · $
                      {e.cost_usd.toFixed(4)}
                    </span>
                  )}
                </span>
              </div>
            );
          })
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
