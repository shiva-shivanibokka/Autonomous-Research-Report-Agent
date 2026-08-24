"use client";

import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import InfoTip from "@/components/InfoTip";

/** Turn a heading into a stable anchor id. */
function slug(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * Split the citation section off the end of the report.
 *
 * The Markdown is rendered server-side and carries its own citation list, but
 * the UI renders citations from the structured array instead — it can group them
 * by domain and give each its own row, which a paragraph of links cannot. Cutting
 * the tail here avoids showing the same thirty sources twice.
 */
function splitCitations(markdown: string): string {
  const i = markdown.search(/\n## Citations \(/);
  return i === -1 ? markdown : markdown.slice(0, i);
}

export default function ReportView({ markdown }: { markdown: string }) {
  const body = useMemo(() => splitCitations(markdown), [markdown]);

  // A ten-section report is long enough that landing at the top with no idea
  // what is below is a worse read than it needs to be.
  const sections = useMemo(
    () =>
      body
        .split("\n")
        .filter((l) => l.startsWith("## "))
        .map((l) => l.slice(3).trim()),
    [body],
  );

  return (
    <div className="panel p-6 sm:p-10">
      <div className="mb-6 flex items-center">
        <span className="eyebrow">Generated report</span>
        <InfoTip text="Written by the Writer agent from the claims the Critic approved, in the structure its report mode defines. Every figure in it traces to a source in the list below; the model was given the claims and the citations, not the open web." />
      </div>

      {sections.length > 2 && (
        <nav className="mb-8 rounded-xl border border-line bg-ink-700/40 p-5">
          <div className="mb-3 flex items-center">
            <span className="font-mono text-[11.5px] uppercase tracking-[0.18em] text-text-faint">
              Contents
            </span>
            <InfoTip text="The sections the Writer chose for this report. They are not a fixed template — the agent decides how to divide the findings, so a different question produces a different structure." />
          </div>
          <ol className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
            {sections.map((s, i) => (
              <li key={s} className="flex gap-2.5">
                <span className="font-mono text-[13px] text-text-faint">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <a
                  href={`#${slug(s)}`}
                  className="text-[15.5px] leading-7 text-text-muted transition hover:text-signal"
                >
                  {s}
                </a>
              </li>
            ))}
          </ol>
        </nav>
      )}

      <article className="report">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h2: ({ children }) => (
              <h2 id={slug(String(children))}>{children}</h2>
            ),
          }}
        >
          {body}
        </ReactMarkdown>
      </article>
    </div>
  );
}
