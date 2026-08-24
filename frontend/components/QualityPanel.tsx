"use client";

import InfoTip from "@/components/InfoTip";
import type { Quality } from "@/lib/types";

function Meter({
  label,
  value,
  invert = false,
  tip,
}: {
  label: string;
  value: number;
  invert?: boolean;
  tip: string;
}) {
  const pct = Math.round(value * 100);
  // For "good = high" metrics, tint by score. Contradiction rate inverts.
  const good = invert ? value <= 0.25 : value >= 0.6;
  const mid = invert ? value <= 0.5 : value >= 0.4;
  const tone = good ? "bg-verified" : mid ? "bg-signal" : "bg-alert";
  const text = good ? "text-verified" : mid ? "text-signal" : "text-alert";
  return (
    <div className="rounded-xl border border-line bg-ink-700/40 p-4">
      <div className="flex items-center">
        <span className="font-mono text-[11.5px] uppercase tracking-[0.14em] text-text-muted">
          {label}
        </span>
        <InfoTip text={tip} />
      </div>
      <div className={`mt-2 font-display text-[2rem] font-semibold ${text}`}>
        {pct}%
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-ink">
        <div
          className={`h-full rounded-full ${tone} transition-[width] duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tip,
}: {
  label: string;
  value: number | string;
  tip: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-ink-700/40 px-4 py-3.5">
      <div className="font-display text-[1.7rem] font-semibold leading-none text-text">
        {value}
      </div>
      <div className="mt-1.5 flex items-center font-mono text-[11.5px] uppercase tracking-[0.12em] text-text-faint">
        {label}
        <InfoTip text={tip} />
      </div>
    </div>
  );
}

// Ordered strongest to weakest so the bar reads left to right as evidence quality.
const LEVELS = [
  { key: "high", label: "high", bar: "bg-verified", dot: "bg-verified" },
  { key: "medium", label: "medium", bar: "bg-signal", dot: "bg-signal" },
  { key: "low", label: "low", bar: "bg-text-faint", dot: "bg-text-faint" },
  { key: "contested", label: "contested", bar: "bg-alert", dot: "bg-alert" },
  {
    key: "inconclusive",
    label: "inconclusive",
    bar: "bg-line",
    dot: "bg-line",
  },
] as const;

export default function QualityPanel({ quality }: { quality: Quality }) {
  const conf = quality.confidence_distribution || {};
  const total = LEVELS.reduce((n, l) => n + (conf[l.key] ?? 0), 0);

  return (
    <div className="panel p-6 sm:p-8">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <span className="flex items-center">
          <span className="eyebrow">Quality assessment</span>
          <InfoTip text="The Critic's scores for this report, carried in the output as data rather than prose. They are what a reader needs to judge how much weight the report deserves — and low numbers are reported rather than hidden, which is the entire point of scoring a report at all." />
        </span>
        <span
          className={`rounded-full border px-3 py-1 font-mono text-[11px] uppercase tracking-wider ${
            quality.converged
              ? "border-verified/50 bg-verified/10 text-verified"
              : "border-signal/50 bg-signal/10 text-signal"
          }`}
        >
          {quality.converged ? "converged" : "did not converge"}
        </span>
      </div>

      {/* Four scores across the width rather than stacked in a narrow column. */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Meter
          label="Overall"
          value={quality.overall_quality_score}
          tip="The Critic's summary judgement, weighing coverage against source diversity. Treat it as the headline: a report scoring below about 60% is worth reading with the contradiction map open beside it."
        />
        <Meter
          label="Coverage"
          value={quality.coverage_score}
          tip="How much of the decomposed question actually got answered. The Orchestrator splits the query into sub-questions; this is the fraction the evidence genuinely addressed, not the fraction attempted."
        />
        <Meter
          label="Source diversity"
          value={quality.source_diversity_score}
          tip="How spread across domains the evidence is. Many citations from one site is weak triangulation however confident the prose sounds — the Sources panel below shows the actual domain breakdown."
        />
        <Meter
          label="Contradiction rate"
          value={quality.contradiction_rate}
          invert
          tip="The share of claims where sources disagree. Inverted: low is good. A non-zero rate is not a defect — it means the pipeline found real disagreement in the literature and said so rather than smoothing it over."
        />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1.15fr]">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat
            label="Sources"
            value={quality.total_sources_consulted}
            tip="Unique URLs cited in the final report, accumulated across every research round."
          />
          <Stat
            label="Claims"
            value={quality.total_claims_extracted}
            tip="Individual factual claims the Analyst agents extracted from the scraped pages, each with its own supporting and contradicting source counts."
          />
          <Stat
            label="Verified"
            value={quality.claims_verified_by_fact_checker}
            tip="Claims a Fact-Checker independently confirmed with a fresh search. Zero means the Critic flagged nothing for re-checking, not that verification failed."
          />
          <Stat
            label="Rounds"
            value={quality.re_research_rounds}
            tip="Research rounds run. More than one means the Critic judged the first pass insufficient and sent the pipeline back for another attempt."
          />
        </div>

        {/* Confidence spread as one bar — the shape of the evidence at a glance. */}
        <div className="rounded-xl border border-line bg-ink-700/40 p-4">
          <div className="flex items-center">
            <span className="font-mono text-[11.5px] uppercase tracking-[0.14em] text-text-muted">
              Confidence distribution
            </span>
            <InfoTip
              align="right"
              text="Every claim scored by source triangulation: high means three or more independent sources agree, medium two, low a single source, contested means sources directly conflict. A report weighted towards 'low' is built on thin evidence, and this is where you can see that."
            />
          </div>

          {total > 0 ? (
            <>
              <div className="mt-3 flex h-3 overflow-hidden rounded-full bg-ink">
                {LEVELS.map((l) => {
                  const n = conf[l.key] ?? 0;
                  if (n === 0) return null;
                  return (
                    <div
                      key={l.key}
                      className={`${l.bar} transition-[width] duration-700`}
                      style={{ width: `${(n / total) * 100}%` }}
                      title={`${l.label}: ${n}`}
                    />
                  );
                })}
              </div>
              <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 font-mono text-[12.5px]">
                {LEVELS.map((l) => {
                  const n = conf[l.key] ?? 0;
                  if (n === 0) return null;
                  return (
                    <span key={l.key} className="flex items-center gap-2">
                      <span className={`h-2 w-2 rounded-sm ${l.dot}`} />
                      <span className="text-text-muted">{l.label}</span>
                      <span className="text-text">{n}</span>
                    </span>
                  );
                })}
              </div>
            </>
          ) : (
            <p className="mt-3 font-mono text-[12.5px] text-text-faint">
              No claims scored.
            </p>
          )}
        </div>
      </div>

      {quality.convergence_note && (
        <p className="mt-5 border-t border-line pt-4 text-[15px] leading-7 text-text-muted">
          {quality.convergence_note}
        </p>
      )}
    </div>
  );
}
