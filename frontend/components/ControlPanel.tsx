"use client";

import { useState } from "react";
import { listModels, type GenerateArgs } from "@/lib/api";
import type { Provider, ReportMode } from "@/lib/types";

const PROVIDERS: { value: Provider; label: string }[] = [
  { value: "anthropic", label: "Anthropic" },
  { value: "openai", label: "OpenAI" },
  { value: "google", label: "Google Gemini" },
  { value: "groq", label: "Groq" },
];

const DEFAULT_MODEL: Record<Provider, string> = {
  anthropic: "claude-sonnet-4-5",
  openai: "gpt-4o-mini",
  google: "gemini-1.5-flash",
  groq: "llama-3.3-70b-versatile",
};

const MODES: { value: ReportMode; label: string }[] = [
  { value: "general", label: "General research" },
  { value: "competitive_intelligence", label: "Competitive intelligence" },
  { value: "investment_thesis", label: "Investment thesis" },
  { value: "academic_literature_review", label: "Literature review" },
];

export default function ControlPanel({
  onSubmit,
  running,
}: {
  onSubmit: (args: GenerateArgs) => void;
  running: boolean;
}) {
  const [query, setQuery] = useState("");
  const [reportMode, setReportMode] = useState<ReportMode>("general");
  const [provider, setProvider] = useState<Provider>("anthropic");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState(DEFAULT_MODEL.anthropic);
  const [models, setModels] = useState<string[]>([DEFAULT_MODEL.anthropic]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [maxRounds, setMaxRounds] = useState(2);
  const [tokenBudget, setTokenBudget] = useState(80);

  function changeProvider(p: Provider) {
    setProvider(p);
    setModel(DEFAULT_MODEL[p]);
    setModels([DEFAULT_MODEL[p]]);
    setModelError(null);
  }

  async function loadModels() {
    if (!apiKey.trim()) {
      setModelError("Enter your API key first.");
      return;
    }
    setLoadingModels(true);
    setModelError(null);
    try {
      const list = await listModels(provider, apiKey.trim());
      if (list.length) {
        setModels(list);
        setModel(list.includes(model) ? model : list[0]);
      }
    } catch (e) {
      setModelError(e instanceof Error ? e.message : "Could not load models.");
    } finally {
      setLoadingModels(false);
    }
  }

  const canRun = query.trim().length >= 10 && !running;

  function submit() {
    onSubmit({
      query: query.trim(),
      reportMode,
      maxRounds,
      tokenBudget: tokenBudget * 1000,
      provider,
      model,
      apiKey: apiKey.trim() || undefined,
    });
  }

  return (
    <div className="panel p-5 sm:p-6">
      <label className="field-label" htmlFor="q">
        Research question
      </label>
      <textarea
        id="q"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        rows={3}
        placeholder="e.g. What is the competitive landscape for AI coding assistants in 2026?"
        className="input resize-none"
      />

      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <label className="field-label" htmlFor="provider">
            Provider
          </label>
          <select
            id="provider"
            value={provider}
            onChange={(e) => changeProvider(e.target.value as Provider)}
            className="input"
          >
            {PROVIDERS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="field-label" htmlFor="mode">
            Report mode
          </label>
          <select
            id="mode"
            value={reportMode}
            onChange={(e) => setReportMode(e.target.value as ReportMode)}
            className="input"
          >
            {MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-4">
        <label className="field-label" htmlFor="key">
          API key <span className="text-text-faint">(BYOK — sent per request, never stored)</span>
        </label>
        <input
          id="key"
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder={`Your ${PROVIDERS.find((p) => p.value === provider)?.label} key`}
          autoComplete="off"
          className="input font-mono"
        />
      </div>

      <div className="mt-4">
        <label className="field-label" htmlFor="model">
          Model
        </label>
        <div className="flex gap-2">
          <select
            id="model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="input"
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={loadModels}
            disabled={loadingModels}
            className="shrink-0 rounded-lg border border-line bg-ink-700 px-3 font-mono text-[11px] uppercase tracking-wider text-text-muted transition hover:border-signal/50 hover:text-text disabled:opacity-50"
          >
            {loadingModels ? "…" : "Load"}
          </button>
        </div>
        {modelError && (
          <p className="mt-1.5 font-mono text-[11px] text-alert">{modelError}</p>
        )}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <label className="field-label" htmlFor="rounds">
            Max rounds <span className="text-text">{maxRounds}</span>
          </label>
          <input
            id="rounds"
            type="range"
            min={1}
            max={3}
            value={maxRounds}
            onChange={(e) => setMaxRounds(Number(e.target.value))}
            className="w-full accent-signal"
          />
        </div>
        <div>
          <label className="field-label" htmlFor="budget">
            Token budget <span className="text-text">{tokenBudget}k</span>
          </label>
          <input
            id="budget"
            type="range"
            min={20}
            max={200}
            step={10}
            value={tokenBudget}
            onChange={(e) => setTokenBudget(Number(e.target.value))}
            className="w-full accent-signal"
          />
        </div>
      </div>

      <button
        type="button"
        onClick={submit}
        disabled={!canRun}
        className="mt-5 w-full rounded-lg border border-signal/60 bg-signal/15 py-3 font-display text-sm font-semibold uppercase tracking-[0.14em] text-signal shadow-glow transition hover:bg-signal/25 disabled:cursor-not-allowed disabled:border-line disabled:bg-ink-700 disabled:text-text-faint disabled:shadow-none"
      >
        {running ? "Researching…" : "Run research"}
      </button>
    </div>
  );
}
