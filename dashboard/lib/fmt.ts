// Tiny formatting helpers shared across log panels.
// No hooks here — pure functions.

import type { EventMsg } from "./api";

/** HH:MM:SS from a Date.now() timestamp. */
export function fmtClock(ms: number): string {
  const d = new Date(ms);
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(d.getMinutes()).padStart(2, "0");
  const s = String(d.getSeconds()).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

/** One-line human-readable summary of an event. Mirrors the Python
 *  `_fmt_event` in main.py but keeps only the useful-for-dashboard bits. */
export function fmtEvent(ev: EventMsg): string {
  const p = ev.payload;
  const tid = ev.track_id !== null ? `id=${ev.track_id}` : "";
  const zones =
    ev.zones && ev.zones.length > 0 ? `[${ev.zones.join(",")}]` : "";
  switch (ev.type) {
    case "new_person":
      return `${tid} ${zones} pose=${p.initial_pose ?? "?"}`.trim();
    case "lost_person":
      return `${tid} last=${
        Array.isArray(p.last_zones) ? `[${(p.last_zones as string[]).join(",")}]` : "?"
      } missed=${p.frames_missing ?? "?"}f`.trim();
    case "pose_change":
      return `${tid} ${p.from_pose ?? "?"} → ${p.to_pose ?? "?"} ${zones}`.trim();
    case "zone_transition": {
      const from = Array.isArray(p.from_zones)
        ? `[${(p.from_zones as string[]).join(",")}]`
        : "?";
      const to = Array.isArray(p.to_zones)
        ? `[${(p.to_zones as string[]).join(",")}]`
        : "?";
      return `${tid} ${from} → ${to}`.trim();
    }
    case "unusual_sound_class":
      return `${p.class_name ?? "?"} conf=${Number(p.confidence ?? 0).toFixed(2)} ${
        Number(p.db_level ?? 0).toFixed(0)
      }dB`;
    case "audio_spike":
      return `current=${Number(p.current_db ?? 0).toFixed(0)}dB baseline=${Number(
        p.baseline_db ?? 0
      ).toFixed(0)}dB Δ=${Number(p.delta_db ?? 0).toFixed(0)}dB`;
    case "speech_start":
      return `conf=${Number(p.confidence ?? 0).toFixed(2)} ${Number(
        p.db_level ?? 0
      ).toFixed(0)}dB`;
    case "speech_end":
      return `duration=${Number(p.duration_seconds ?? 0).toFixed(1)}s`;
    default:
      return JSON.stringify(p).slice(0, 120);
  }
}

/** Which sensor produced the event — drives the row accent color in
 *  PerceptionLog. Keep in sync with Layer 0 event types. */
export type EventSource = "vision" | "audio" | "power";

export function eventSource(type: string): EventSource {
  if (
    type === "new_person" ||
    type === "lost_person" ||
    type === "pose_change" ||
    type === "zone_transition"
  )
    return "vision";
  if (
    type === "unusual_sound_class" ||
    type === "audio_spike" ||
    type === "speech_start" ||
    type === "speech_end"
  )
    return "audio";
  return "power";
}

/** Tailwind left-border class per source, for row accent strips. */
export const EVENT_SOURCE_ACCENT: Record<EventSource, string> = {
  vision: "border-l-emerald-500",
  audio: "border-l-violet-500",
  power: "border-l-amber-500",
};

/** Event type → Tailwind class for the badge color. */
export const EVENT_COLORS: Record<string, string> = {
  new_person: "bg-emerald-700 text-emerald-100",
  lost_person: "bg-rose-800 text-rose-100",
  pose_change: "bg-amber-800 text-amber-100",
  zone_transition: "bg-sky-800 text-sky-100",
  unusual_sound_class: "bg-fuchsia-800 text-fuchsia-100",
  audio_spike: "bg-orange-800 text-orange-100",
  speech_start: "bg-yellow-900 text-yellow-100",
  speech_end: "bg-yellow-900 text-yellow-100",
  beat_1: "bg-zinc-700 text-zinc-100",
  beat_2: "bg-indigo-700 text-indigo-100",
};
