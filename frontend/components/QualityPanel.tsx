"use client";

import InfoTip from "@/components/InfoTip";
import type { Quality } from "@/lib/types";

function Meter({
  label,
  value,
  invert = false,
}: {
  label: string;
  value: number;
  invert?: boolean;
}) {
  const pct = Math.round(value * 100);
  // For "good = high" metrics, tint by score. Contradiction rate inverts.
  const good = invert ? value <= 0.25 : value >= 0.6;
  const mid = invert ? value <= 0.5 : value >= 0.4;
  const tone = good ? "bg-verified" : mid ? "bg-signal" : "bg-alert";
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="font-mono text-[12px] uppercase tracking-wider text-text-muted">
          {label}
        </span>
        <span className="font-mono text-sm text-text">{pct}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-ink-700">
        <div
          className={`h-full rounded-full ${tone} transition-[width] duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-lg border border-line bg-ink-700/50 px-3 py-2.5">
      <div className="font-display text-xl font-semibold text-text">{value}</div>
      <div className="mt-0.5 font-mono text-[11px] uppercase tracking-wider text-text-faint">
        {label}
      </div>
    </div>
  );
}

export default function QualityPanel({ quality }: { quality: Quality }) {
  const conf = quality.confidence_distribution || {};
  return (
    <div className="panel p-6">
      <div className="mb-4 flex items-center justify-between">
        <span className="flex items-center">
          <span className="eyebrow">Quality assessment</span>
          <InfoTip text="The Critic's scores for this report, carried in the output as data rather than prose. Coverage is how much of the decomposed question got answered; source diversity is how spread across domains the evidence is; contradiction rate is the share of claims where sources disagree. Low numbers are reported, not hidden — that is the point of scoring a report at all." />
        </span>
        <span
          className={`font-mono text-[11px] ${
            quality.converged ? "text-verified" : "text-signal"
          }`}
        >
          {quality.converged ? "converged" : "budget-limited"}
        </span>
      </div>

      <div className="space-y-3.5">
        <Meter label="Overall" value={quality.overall_quality_score} />
        <Meter label="Coverage" value={quality.coverage_score} />
        <Meter label="Source diversity" value={quality.source_diversity_score} />
        <Meter
          label="Contradiction rate"
          value={quality.contradiction_rate}
          invert
        />
      </div>

      <div className="mt-5 grid grid-cols-3 gap-2">
        <Stat label="Sources" value={quality.total_sources_consulted} />
        <Stat label="Claims" value={quality.total_claims_extracted} />
        <Stat label="Verified" value={quality.claims_verified_by_fact_checker} />
      </div>

      <div className="mt-4">
        <div className="mb-2 font-mono text-[12px] uppercase tracking-wider text-text-muted">
          Confidence distribution
        </div>
        <div className="flex flex-wrap gap-1.5 font-mono text-[12px]">
          {(["high", "medium", "low", "contested", "inconclusive"] as const).map(
            (k) =>
              (conf[k] ?? 0) > 0 ? (
                <span
                  key={k}
                  className="rounded border border-line bg-ink-700/60 px-2 py-1 text-text-muted"
                >
                  {k} <span className="text-text">{conf[k]}</span>
                </span>
              ) : null,
          )}
        </div>
      </div>

      {quality.convergence_note && (
        <p className="mt-4 border-t border-line pt-3 text-[14px] leading-7 text-text-muted">
          {quality.convergence_note}
        </p>
      )}
    </div>
  );
}
