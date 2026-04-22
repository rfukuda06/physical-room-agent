// PerceptionLog — Layer 0 sensor stream. Not an LLM agent (no model, no
// reasoning, no cost) but it IS the system's continuous nervous system:
// YOLO's person detection + tracking + pose, YAMNet's audio classifier,
// and the event detector that stitches raw samples into semantic events.
//
// Styled differently from the agent panels: emerald accent (live signal),
// a "LIVE" heartbeat pulse to convey continuous operation, and model
// chips naming the underlying networks. Each row gets a source-colored
// left strip (emerald = vision, violet = audio, amber = power) so you
// can scan the stream and see where activity is coming from at a glance.

"use client";

import { useEffect, useState } from "react";
import type { EventMsg } from "@/lib/api";
import {
  EVENT_COLORS,
  EVENT_SOURCE_ACCENT,
  eventSource,
  fmtClock,
  fmtEvent,
} from "@/lib/fmt";

type Entry = EventMsg & { _localId: number; _receivedAt: number };

type Props = {
  events: Entry[];
  className?: string;
};

export default function PerceptionLog({ events, className = "" }: Props) {
  // Tick every second so the "events/10s" rate decays smoothly when
  // activity dies off, not only when a new event arrives.
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const reversed = events.slice().reverse();

  // Tally by source for the header — small telemetry that conveys
  // "multiple modalities, all running."
  const counts = { vision: 0, audio: 0, power: 0 };
  for (const e of events) counts[eventSource(e.type)]++;

  // Rough activity rate: events in the last 10s. Gives the "pulse" a
  // meaning beyond decoration.
  const now = Date.now();
  const recent = events.filter((e) => now - e._receivedAt < 10_000).length;

  return (
    <div
      className={`flex min-h-0 flex-col overflow-hidden rounded-lg border border-l-[3px] border-zinc-800 border-l-emerald-500 bg-zinc-900/60 ${className}`}
    >
      {/* Header: live indicator + title + model chips + source tallies. */}
      <div className="relative flex shrink-0 items-center gap-3 border-b border-zinc-800 bg-gradient-to-r from-emerald-500/15 to-transparent px-3 py-2">
        {/* Heartbeat — always on, pulsing to signal continuous Layer 0. */}
        <div className="relative flex h-9 w-9 shrink-0 items-center justify-center">
          <span className="absolute inline-flex h-9 w-9 animate-ping rounded-full bg-emerald-500/30" />
          <span className="relative inline-flex h-3 w-3 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981]" />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-1">
            <h3 className="truncate text-sm font-semibold text-zinc-100">
              Perception
            </h3>
            <div className="flex min-w-0 flex-wrap gap-1">
              <ModelChip label="YOLO26n-pose" />
              <ModelChip label="YAMNet" />
            </div>
          </div>
          <div className="truncate text-[10px] text-zinc-500">
            Layer 0 — local, continuous, no LLM
          </div>
        </div>

        <div className="flex shrink-0 flex-col items-end gap-1 text-[10px]">
          <div className="flex items-center gap-1.5">
            <span className="font-medium text-emerald-400">
              {recent}/10s
            </span>
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-zinc-400">
              {events.length}
            </span>
          </div>
          <div className="flex gap-1 font-mono text-[10px]">
            <SourceTally color="text-emerald-400" label="vis" n={counts.vision} />
            <SourceTally color="text-violet-400" label="aud" n={counts.audio} />
            {counts.power > 0 && (
              <SourceTally
                color="text-amber-400"
                label="pwr"
                n={counts.power}
              />
            )}
          </div>
        </div>
      </div>

      {/* Body — no scroll. Newest at top. */}
      <div className="min-h-0 flex-1 overflow-hidden px-3 py-2">
        {events.length === 0 ? (
          <div className="py-6 text-center text-sm text-zinc-500">
            listening…
          </div>
        ) : (
          <ul className="space-y-1 font-mono text-[12px]">
            {reversed.map((e) => {
              const src = eventSource(e.type);
              return (
                <li
                  key={e._localId}
                  className={`flex items-start gap-2 rounded border border-zinc-800/50 border-l-2 bg-zinc-950/40 px-2 py-1 ${EVENT_SOURCE_ACCENT[src]}`}
                >
                  <span className="shrink-0 text-zinc-500">
                    {fmtClock(e._receivedAt)}
                  </span>
                  <span
                    className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                      EVENT_COLORS[e.type] ?? "bg-zinc-700 text-zinc-100"
                    }`}
                  >
                    {e.type}
                  </span>
                  <span className="truncate text-zinc-300">{fmtEvent(e)}</span>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------

function ModelChip({ label }: { label: string }) {
  return (
    <span className="rounded border border-emerald-900/70 bg-emerald-950/60 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-emerald-300">
      {label}
    </span>
  );
}

function SourceTally({
  color,
  label,
  n,
}: {
  color: string;
  label: string;
  n: number;
}) {
  return (
    <span className="flex items-center gap-0.5">
      <span className={color}>{label}</span>
      <span className="text-zinc-500">{n}</span>
    </span>
  );
}
