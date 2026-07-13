import type { Config } from "tailwindcss";

// "Research instrument / control room" tokens. Deep ink base (not pure black),
// a single warm signal-amber for processing, cyan reserved for verified/complete.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#0B0F17", // page background
          800: "#131A26", // surface
          700: "#1A2233", // raised surface
          600: "#212C42",
        },
        line: "#26314A", // hairline borders
        text: {
          DEFAULT: "#E7EDF7",
          muted: "#8A97AE",
          faint: "#5C6880",
        },
        signal: "#FFB020", // amber — active / processing
        verified: "#3FD0C9", // cyan — complete / verified
        alert: "#F0616D", // red — failed
      },
      fontFamily: {
        display: ["var(--font-display)", "system-ui", "sans-serif"],
        sans: ["var(--font-body)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 30px -12px rgba(0,0,0,0.6)",
        glow: "0 0 0 1px rgba(255,176,32,0.35), 0 0 24px -6px rgba(255,176,32,0.4)",
      },
      keyframes: {
        pulseSignal: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
        sweep: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        rise: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "pulse-signal": "pulseSignal 1.4s ease-in-out infinite",
        sweep: "sweep 1.6s linear infinite",
        rise: "rise 0.25s ease-out both",
      },
    },
  },
  plugins: [],
};

export default config;
