"use client";

import { useMemo, useState } from "react";
import InfoTip from "@/components/InfoTip";

export interface Citation {
  index: number;
  url: string;
  title: string;
  domain: string;
  accessed_date: string;
}

/**
 * The source list, rendered from the structured citation array.
 *
 * These used to come through the report's Markdown as one plain line each, and
 * Markdown folds consecutive lines into a single paragraph — so thirty sources
 * arrived as one run-on block that was effectively unreadable. Rendering from
 * the data instead of the prose also makes the useful view possible: grouping by
 * domain shows at a glance whether the research leaned on one site, which is
 * exactly what the source-diversity score is measuring.
 */
export default function Citations({ citations }: { citations: Citation[] }) {
  const [byDomain, setByDomain] = useState(false);

  const domains = useMemo(() => {
    const map = new Map<string, Citation[]>();
    for (const c of citations) {
      const key = c.domain || new URL(c.url).hostname;
      const list = map.get(key);
      if (list) list.push(c);
      else map.set(key, [c]);
    }
    // Most-used first: a domain contributing six sources is the story here.
    return [...map.entries()].sort(
      (a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]),
    );
  }, [citations]);

  if (citations.length === 0) return null;

  return (
    <div className="panel p-6 sm:p-8">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center">
          <span className="eyebrow">
            Sources · {citations.length} cited · {domains.length} domains
          </span>
          <InfoTip text="Every source the report draws on, numbered as it is cited in the text. Grouping by domain shows how concentrated the research is: many citations from one site is weak triangulation, however confident the prose sounds — it is what the source-diversity score measures." />
        </div>
        <button
          onClick={() => setByDomain((v) => !v)}
          className="rounded-md border border-line px-3 py-1.5 font-mono text-[12px] uppercase tracking-wider text-text-muted transition hover:border-signal/50 hover:text-signal"
        >
          {byDomain ? "list in order" : "group by domain"}
        </button>
      </div>

      {byDomain ? (
        <div className="space-y-5">
          {domains.map(([domain, items]) => (
            <div key={domain}>
              <div className="mb-2 flex items-baseline gap-2">
                <span className="font-mono text-[13px] text-verified">{domain}</span>
                <span className="font-mono text-[11.5px] text-text-faint">
                  {items.length} {items.length === 1 ? "source" : "sources"}
                </span>
              </div>
              <ol className="space-y-2">
                {items.map((c) => (
                  <CitationRow key={c.index} c={c} />
                ))}
              </ol>
            </div>
          ))}
        </div>
      ) : (
        <ol className="space-y-2.5">
          {citations.map((c) => (
            <CitationRow key={c.index} c={c} showDomain />
          ))}
        </ol>
      )}
    </div>
  );
}

function CitationRow({ c, showDomain }: { c: Citation; showDomain?: boolean }) {
  return (
    <li className="flex gap-3 border-l-2 border-line pl-3 transition hover:border-signal/50">
      <span className="mt-0.5 shrink-0 font-mono text-[13px] text-text-faint">
        [{c.index}]
      </span>
      <span className="min-w-0">
        <a
          href={c.url}
          target="_blank"
          rel="noreferrer"
          className="text-[16px] leading-7 text-text underline decoration-line underline-offset-4 transition hover:decoration-signal"
        >
          {c.title || c.url}
        </a>
        <span className="mt-0.5 block break-all font-mono text-[12.5px] leading-6 text-text-faint">
          {showDomain && c.domain ? `${c.domain} · ` : ""}
          {c.url}
        </span>
      </span>
    </li>
  );
}
