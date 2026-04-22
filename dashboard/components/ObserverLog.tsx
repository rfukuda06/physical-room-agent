// ObserverLog — past Observer (Gemini) narrations.
// Themed in Gemini blue via AgentPanel.

"use client";

import type { NarrationMsg } from "@/lib/api";
import { fmtClock } from "@/lib/fmt";
import AgentPanel, { OBSERVER_THEME } from "./AgentPanel";

type Entry = NarrationMsg & { _localId: number; _receivedAt: number };

type Props = {
  narrations: Entry[];
  model?: string;
  className?: string;
};

export default function ObserverLog({ narrations, className }: Props) {
  const reversed = narrations.slice().reverse();
  const latest = narrations.at(-1);

  return (
    <AgentPanel
      theme={OBSERVER_THEME}
      count={narrations.length}
      lastActivityMs={latest?._receivedAt}
      className={className}
    >
      <ul className="space-y-1.5 text-sm">
        {reversed.map((n) => {
          // Rows that landed in the last 5s glow in the agent's brand
          // color, then fade back to the static row style. Computed at
          // render time; CSS animation is one-shot so it plays once on
          // mount and doesn't re-fire on subsequent re-renders.
          const isRecent = Date.now() - n._receivedAt < 5000;
          return (
          <li
            key={n._localId}
            className="rounded border border-zinc-800/50 border-l-2 border-l-sky-500/70 bg-zinc-950/40 p-2"
            style={
              isRecent
                ? { animation: "row-highlight-sky 5000ms ease-out" }
                : undefined
            }
          >
            <div className="mb-1 flex flex-wrap items-center gap-1.5 text-[10px]">
              <span className="font-mono text-zinc-500">
                {fmtClock(n._receivedAt)}
              </span>
              {n.escalate ? (
                <span className="rounded bg-amber-800 px-1.5 py-0.5 font-medium text-amber-100">
                  escalated
                </span>
              ) : (
                <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">
                  triaged
                </span>
              )}
              {n.trigger_events && n.trigger_events.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {n.trigger_events.map((t, i) => (
                    <span
                      key={`${n._localId}-${i}`}
                      className="rounded bg-sky-950/70 px-1.5 py-0.5 font-mono text-sky-300"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div className="text-zinc-200">{n.narration || "(empty)"}</div>
            {n.escalate && n.escalate_reason && (
              <div className="mt-1 text-[11px] italic text-zinc-500">
                {n.escalate_reason}
              </div>
            )}
          </li>
          );
        })}
      </ul>
    </AgentPanel>
  );
}
