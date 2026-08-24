"use client";

import { useEffect, useId, useRef, useState } from "react";

/**
 * A small "?" that explains what a panel is actually showing.
 *
 * Opens on hover for a mouse, on focus for a keyboard, and on tap for touch —
 * a hover-only tooltip is invisible on a phone, which is where a fair number of
 * people will open a portfolio link. Closes on Escape or an outside tap.
 */
export default function InfoTip({
  text,
  align = "left",
}: {
  text: string;
  align?: "left" | "right";
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLSpanElement>(null);
  const id = useId();

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    function onDown(e: PointerEvent) {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
    };
  }, [open]);

  return (
    <span ref={wrap} className="relative ml-2 inline-flex items-center align-middle">
      <button
        type="button"
        aria-label="What is this?"
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onClick={() => setOpen((v) => !v)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="grid h-[18px] w-[18px] place-items-center rounded-full border border-line bg-ink-700 font-mono text-[11px] leading-none text-text-faint transition hover:border-signal/60 hover:text-signal focus:border-signal focus:text-signal focus:outline-none"
      >
        ?
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className={`absolute top-[26px] z-30 w-[min(22rem,70vw)] rounded-lg border border-line bg-ink p-3 text-[13.5px] font-normal normal-case leading-6 tracking-normal text-text-muted shadow-panel ${
            align === "right" ? "right-0" : "left-0"
          }`}
        >
          {text}
        </span>
      )}
    </span>
  );
}
