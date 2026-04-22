// AgentPanel — shared chrome for the Observer and Reasoner cards.
//
// The panel itself is static: brand-colored left strip, subtle header
// gradient, avatar. No more whole-panel lighting. When the agent
// "speaks", the status pill in the header flips to SPEAKING in the
// brand color — that's the only chrome indicator. The new narration
// row inside the body lights up on its own (see row-highlight-sky /
// row-highlight-orange keyframes in globals.css).

"use client";

import { useEffect, useState } from "react";

// Inline SVGs for the two agent avatars. Stroked with currentColor so
// the avatar can retint them; designed on a 24×24 viewBox so Tailwind
// `h-5 w-5` sizing inside the 36px disc lands on a nice optical size.
function EyeIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-5 w-5"
      aria-hidden="true"
    >
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" fill="currentColor" />
    </svg>
  );
}

function BrainIcon() {
  // Simplified two-hemisphere brain: each side is a rounded lobe with
  // one curl, meeting at the vertical seam in the middle.
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-5 w-5"
      aria-hidden="true"
    >
      <path d="M9.5 3.5a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0-2 3.6A2.5 2.5 0 0 0 4 12a2.5 2.5 0 0 0 1.1 2.1 2.5 2.5 0 0 0 .4 3A2.5 2.5 0 0 0 7.5 20.5a2.5 2.5 0 0 0 2 1 2.5 2.5 0 0 0 2.5-2.5V5.5a2 2 0 0 0-2.5-2z" />
      <path d="M14.5 3.5a2.5 2.5 0 0 1 2.5 2.5 2.5 2.5 0 0 1 2 3.6A2.5 2.5 0 0 1 20 12a2.5 2.5 0 0 1-1.1 2.1 2.5 2.5 0 0 1-.4 3A2.5 2.5 0 0 1 16.5 20.5a2.5 2.5 0 0 1-2 1 2.5 2.5 0 0 1-2.5-2.5V5.5a2 2 0 0 1 2.5-2z" />
      <path d="M9 10h1.5M13.5 10H15M9 14h1.5M13.5 14H15" />
    </svg>
  );
}

export type AgentTheme = {
  role: string;                 // "Observer Agent" / "Reasoner Agent"
  model: string;                // "Gemini 2.5 Flash" / "Claude Sonnet 4.6"
  tagline: string;              // short descriptor under the name
  icon: React.ComponentType;    // avatar glyph
  hex: string;                  // brand color hex
  rgb: string;                  // "r, g, b" triplet for rgba templates
  borderClass: string;          // left accent strip class
  glowClass: string;            // bg gradient stop class for the header
  avatarGradient: string;       // Tailwind avatar gradient stops
};

export const OBSERVER_THEME: AgentTheme = {
  role: "Observer Agent",
  model: "Gemini 2.5 Flash",
  tagline: "Fast factual description",
  icon: EyeIcon,
  hex: "#4285F4",
  rgb: "56, 189, 248",
  borderClass: "border-l-sky-500",
  glowClass: "from-sky-500/20",
  avatarGradient: "from-sky-400 via-blue-500 to-indigo-600",
};

export const REASONER_THEME: AgentTheme = {
  role: "Reasoner Agent",
  model: "Claude Sonnet 4.6",
  tagline: "Judgment, actions, voice",
  icon: BrainIcon,
  hex: "#CC785C",
  rgb: "251, 146, 60",
  borderClass: "border-l-orange-500",
  glowClass: "from-orange-500/20",
  avatarGradient: "from-amber-400 via-orange-500 to-rose-500",
};

// How long the header status pill stays in the "SPEAKING" state after
// a narration. Matches the row-highlight keyframe duration.
const ACTIVE_WINDOW_MS = 5000;

export default function AgentPanel({
  theme,
  count,
  accent,
  lastActivityMs,
  children,
  className = "",
}: {
  theme: AgentTheme;
  count: number;
  accent?: React.ReactNode;
  /** Timestamp (Date.now() ms) of most recent narration. */
  lastActivityMs?: number;
  children: React.ReactNode;
  className?: string;
}) {
  // Re-render every second so the status pill flips from SPEAKING back
  // to the "Xs ago" timer without needing new messages to trigger it.
  const [, setNow] = useState(0);
  useEffect(() => {
    if (!lastActivityMs) return;
    const id = setInterval(() => setNow((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [lastActivityMs]);

  const sinceMs = lastActivityMs ? Date.now() - lastActivityMs : Infinity;
  const isActive = sinceMs < ACTIVE_WINDOW_MS;

  return (
    <div
      className={`flex min-h-0 flex-col overflow-hidden rounded-lg border border-l-[3px] border-zinc-800 bg-zinc-900/60 ${theme.borderClass} ${className}`}
    >
      <div
        className={`relative flex shrink-0 items-center gap-3 border-b border-zinc-800 bg-gradient-to-r to-transparent px-3 py-2 ${theme.glowClass}`}
      >
        {/* Avatar — static; brand gradient disc with the agent glyph
            (eye for Observer, brain for Reasoner). Icons inherit
            currentColor, so the white text color on this div paints
            them white against the brand-gradient background. */}
        <div
          className={`relative flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br text-white shadow-lg ${theme.avatarGradient}`}
          style={{ boxShadow: `0 0 3px ${theme.hex}60` }}
        >
          <theme.icon />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <h3 className="truncate text-sm font-semibold text-zinc-100">
              {theme.role}
            </h3>
            <span className="truncate text-[11px] font-medium text-zinc-400">
              {theme.model}
            </span>
          </div>
          <div className="truncate text-[10px] text-zinc-500">
            {theme.tagline}
          </div>
        </div>

        {/* Status pill — this is the only "active" signal left on the
            panel chrome. SPEAKING in brand color while a narration is
            recent; otherwise "Xs ago" or idle. */}
        <div className="flex shrink-0 flex-col items-end gap-1 text-[10px]">
          <div className="flex items-center gap-1.5">
            {isActive ? (
              <>
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{
                    backgroundColor: theme.hex,
                    boxShadow: `0 0 8px ${theme.hex}`,
                  }}
                />
                <span
                  className="font-semibold uppercase tracking-wider"
                  style={{ color: theme.hex }}
                >
                  speaking
                </span>
              </>
            ) : (
              <>
                <span className="h-1.5 w-1.5 rounded-full bg-zinc-700" />
                <span className="text-zinc-500">
                  {lastActivityMs
                    ? `${Math.floor(sinceMs / 1000)}s ago`
                    : "idle"}
                </span>
              </>
            )}
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-zinc-400">
              {count}
            </span>
          </div>
          {accent && <div className="text-zinc-500">{accent}</div>}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden px-3 py-2">
        {count === 0 ? (
          <div className="py-6 text-center text-sm text-zinc-500">
            waiting for first response…
          </div>
        ) : (
          children
        )}
      </div>
    </div>
  );
}
