// ReasonerLog — Claude Beat-2 narrations + routing skips. Themed in
// Anthropic clay-orange via AgentPanel. Fired routings are suppressed
// because a narration arrives for each one; showing both would double up.

"use client";

import type { NarrationMsg, RoutingMsg } from "@/lib/api";
import { fmtClock } from "@/lib/fmt";
import AgentPanel, { REASONER_THEME } from "./AgentPanel";

type NarrEntry = NarrationMsg & { _localId: number; _receivedAt: number };
type RoutEntry = RoutingMsg & { _localId: number; _receivedAt: number };

type UnifiedEntry =
  | { kind: "narration"; data: NarrEntry }
  | { kind: "skip"; data: RoutEntry };

type Props = {
  narrations: NarrEntry[];
  routings: RoutEntry[];
  model?: string;
  className?: string;
};

export default function ReasonerLog({
  narrations,
  routings,
  className,
}: Props) {
  const unified: UnifiedEntry[] = [
    ...narrations.map<UnifiedEntry>((n) => ({ kind: "narration", data: n })),
  ].sort((a, b) => b.data._localId - a.data._localId);

  const latest = narrations.at(-1);

  return (
    <AgentPanel
      theme={REASONER_THEME}
      count={unified.length}
      lastActivityMs={latest?._receivedAt}
      className={className}
    >
      <ul className="space-y-1.5 text-sm">
        {unified.map((entry) =>
          entry.kind === "narration" ? (
            <NarrationRow key={entry.data._localId} n={entry.data} />
          ) : (
            <SkipRow key={entry.data._localId} r={entry.data} />
          )
        )}
      </ul>
    </AgentPanel>
  );
}

// ---------------------------------------------------------------------

function NarrationRow({ n }: { n: NarrEntry }) {
  const hasActions = n.lamp || n.fan;
  // Recent rows glow in orange and fade back to their static look.
  const isRecent = Date.now() - n._receivedAt < 5000;
  return (
    <li
      className="rounded border border-zinc-800/50 border-l-2 border-l-orange-500 bg-orange-950/10 p-2"
      style={
        isRecent
          ? { animation: "row-highlight-orange 5000ms ease-out" }
          : undefined
      }
    >
      <div className="mb-1 flex flex-wrap items-center gap-1.5 text-[10px]">
        <span className="font-mono text-zinc-500">
          {fmtClock(n._receivedAt)}
        </span>
        {n.alert && (
          <span className="rounded bg-rose-800 px-1.5 py-0.5 font-medium text-rose-100">
            ALERT
          </span>
        )}
        {hasActions && (
          <div className="flex items-center gap-1">
            {n.lamp && (
              <span
                className={`rounded px-1.5 py-0.5 font-medium ${
                  n.lamp === "on"
                    ? "bg-emerald-700 text-emerald-100"
                    : "bg-zinc-700 text-zinc-200"
                }`}
              >
                lamp {n.lamp}
              </span>
            )}
            {n.fan && (
              <span
                className={`rounded px-1.5 py-0.5 font-medium ${
                  n.fan === "on"
                    ? "bg-emerald-700 text-emerald-100"
                    : "bg-zinc-700 text-zinc-200"
                }`}
              >
                fan {n.fan}
              </span>
            )}
          </div>
        )}
      </div>
      {n.narration && (
        <div className="text-zinc-100">{n.narration}</div>
      )}
      {n.reasoning && (
        <div className="mt-1 text-[11px] italic text-zinc-500">
          reasoning: {n.reasoning}
        </div>
      )}
    </li>
  );
}

function SkipRow({ r }: { r: RoutEntry }) {
  return (
    <li className="flex items-start gap-2 rounded border border-zinc-800/50 bg-zinc-950/40 px-2 py-1 text-[12px]">
      <span className="font-mono text-zinc-500">
        {fmtClock(r._receivedAt)}
      </span>
      <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">
        skipped
      </span>
      <span className="font-mono text-[11px] text-zinc-500">{r.trigger}</span>
      {r.reason && (
        <span className="truncate italic text-zinc-500">— {r.reason}</span>
      )}
    </li>
  );
}
