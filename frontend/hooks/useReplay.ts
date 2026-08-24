"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  loadDemoRun,
  playbackSchedule,
  type DemoRun,
} from "@/lib/demo";
import type { ActivityEntry } from "@/lib/types";

export interface ReplayState {
  run: DemoRun | null;
  /** Entries revealed so far — grows as playback advances. */
  entries: ActivityEntry[];
  round: number;
  tokens: number;
  cost: number;
  playing: boolean;
  finished: boolean;
  error: string | null;
}

const IDLE: ReplayState = {
  run: null,
  entries: [],
  round: 0,
  tokens: 0,
  cost: 0,
  playing: false,
  finished: false,
  error: null,
};

/**
 * Plays a recorded run back on a compressed timeline.
 *
 * One timeout per frame rather than a single interval: the frames carry their
 * own real timestamps, and scheduling each independently is what preserves the
 * shape of the run — the long pause while five analysts work in parallel stays
 * visibly longer than the quick handoff after it. An even tick would flatten
 * that into something that looks like a progress bar rather than a pipeline.
 */
export function useReplay() {
  const [state, setState] = useState<ReplayState>(IDLE);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const clearTimers = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }, []);

  useEffect(() => clearTimers, [clearTimers]);

  const start = useCallback(async () => {
    clearTimers();
    setState({ ...IDLE, playing: true });

    let run: DemoRun;
    try {
      run = await loadDemoRun();
    } catch (e) {
      setState({
        ...IDLE,
        error: e instanceof Error ? e.message : "Could not load the recording.",
      });
      return;
    }

    setState((s) => ({ ...s, run, playing: true }));

    const schedule = playbackSchedule(run.frames);
    run.frames.forEach((frame, i) => {
      const t = setTimeout(() => {
        setState((s) => ({
          ...s,
          entries: run.frames.slice(0, i + 1).map((f) => f.entry),
          round: frame.round,
          tokens: frame.tokens_used,
          cost: frame.cost_usd,
        }));
      }, schedule[i]);
      timers.current.push(t);
    });

    const done = setTimeout(
      () =>
        setState((s) => ({
          ...s,
          entries: run.activity_log,
          tokens: run.tokens_used,
          cost: run.cost_usd,
          playing: false,
          finished: true,
        })),
      (schedule[schedule.length - 1] ?? 0) + 400,
    );
    timers.current.push(done);
  }, [clearTimers]);

  /** Jump straight to the finished state — for anyone who does not want to wait. */
  const skip = useCallback(async () => {
    clearTimers();
    const run = state.run ?? (await loadDemoRun());
    setState({
      run,
      entries: run.activity_log,
      round: Math.max(run.rounds_run - 1, 0),
      tokens: run.tokens_used,
      cost: run.cost_usd,
      playing: false,
      finished: true,
      error: null,
    });
  }, [clearTimers, state.run]);

  const reset = useCallback(() => {
    clearTimers();
    setState(IDLE);
  }, [clearTimers]);

  return { ...state, start, skip, reset };
}
