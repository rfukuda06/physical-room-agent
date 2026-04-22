// StatsPanel — right-hand sidebar showing the things you want to see
// at a glance: people count, per-entity pose + zones, audio level +
// dominant class, smart plug state, and calibration status.

"use client";

import type { DashboardConfig, WorldSnapshot } from "@/lib/api";

type Props = {
  world: WorldSnapshot | null;
  config: DashboardConfig | null;
};

export default function StatsPanel({ world, config }: Props) {
  // Laid out horizontally at the bottom of the dashboard. Height is
  // content-driven (no h-full) — this is what lets the right column's
  // Reasoner bottom come UP to match the card bottoms, rather than
  // the cards being stretched DOWN to meet a padded wrapper.
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <PeopleCard world={world} />
      <AudioCard world={world} />
      <DevicesCard world={world} />
      <BaselinesCard world={world} config={config} />
    </div>
  );
}

// ---------------------------------------------------------------------

function Card({
  title,
  children,
  accent,
}: {
  title: string;
  children: React.ReactNode;
  accent?: string;
}) {
  // overflow-hidden clips anything that doesn't fit the fixed row height
  // set by the bottom section in page.tsx — this keeps the dashboard
  // locked to one viewport.
  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
      <div className="mb-2 flex shrink-0 items-center justify-between gap-2">
        <h3 className="truncate text-[11px] font-semibold uppercase tracking-wider text-zinc-400">
          {title}
        </h3>
        {accent && (
          <span className="truncate text-[10px] font-medium text-zinc-500">
            {accent}
          </span>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------

function PeopleCard({ world }: { world: WorldSnapshot | null }) {
  const count = world?.people_count ?? 0;
  return (
    <Card title="People in room" accent={`${count} tracked`}>
      <div className="text-4xl font-semibold text-zinc-100">{count}</div>
      <div className="mt-3 space-y-2">
        {(world?.entities ?? []).length === 0 ? (
          <div className="text-sm text-zinc-500">No one visible</div>
        ) : (
          world?.entities.map((e) => (
            <div
              key={e.id}
              className="flex items-center justify-between rounded border border-zinc-800 bg-zinc-950/60 px-2 py-1 text-sm"
            >
              <div className="flex items-center gap-2">
                <span className="inline-block rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-xs text-zinc-300">
                  id {e.id}
                </span>
                <PoseBadge pose={e.pose} />
                {e.zones.length > 0 && (
                  <span className="text-xs text-emerald-400">
                    {e.zones.join(", ")}
                  </span>
                )}
              </div>
              <div className="text-right text-[11px] text-zinc-500">
                <div>{e.seconds_in_frame.toFixed(0)}s</div>
                <div>{e.velocity_px_per_s.toFixed(0)} px/s</div>
              </div>
            </div>
          ))
        )}
      </div>
    </Card>
  );
}

function PoseBadge({ pose }: { pose: string }) {
  const colors: Record<string, string> = {
    standing: "bg-sky-700 text-sky-100",
    sitting: "bg-violet-700 text-violet-100",
    walking: "bg-emerald-700 text-emerald-100",
    unknown: "bg-zinc-700 text-zinc-300",
  };
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${
        colors[pose] ?? colors.unknown
      }`}
    >
      {pose}
    </span>
  );
}

// ---------------------------------------------------------------------

function AudioCard({ world }: { world: WorldSnapshot | null }) {
  const audio = world?.audio;
  const db = audio?.level_db ?? -100;
  // Map dB range [-60, -10] → [0, 100%] for the bar.
  const pct = Math.max(0, Math.min(100, ((db + 60) / 50) * 100));
  const barColor = audio?.recent_spike
    ? "bg-amber-500"
    : audio?.speech_active
    ? "bg-sky-500"
    : "bg-emerald-500";

  return (
    <Card
      title="Audio"
      accent={audio?.dominant_class || "silence"}
    >
      <div className="flex items-baseline gap-2">
        <div className="text-3xl font-semibold text-zinc-100">
          {db.toFixed(0)}
        </div>
        <div className="text-sm text-zinc-400">dB</div>
        {audio?.speech_active && (
          <span className="ml-auto rounded bg-sky-700 px-1.5 py-0.5 text-[11px] font-medium text-sky-100">
            SPEECH
          </span>
        )}
        {audio?.recent_spike && (
          <span className="ml-auto rounded bg-amber-700 px-1.5 py-0.5 text-[11px] font-medium text-amber-100">
            SPIKE
          </span>
        )}
      </div>

      <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
        <div
          className={`h-full transition-all duration-150 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {audio && audio.top_classes.length > 0 && (
        <div className="mt-3 space-y-1">
          {audio.top_classes.slice(0, 4).map((c) => (
            <div key={c.label} className="flex items-center gap-2">
              <div className="w-24 truncate text-xs text-zinc-400">
                {c.label}
              </div>
              <div className="h-1 flex-1 overflow-hidden rounded bg-zinc-800">
                <div
                  className="h-full bg-zinc-400"
                  style={{ width: `${Math.min(100, c.confidence * 100)}%` }}
                />
              </div>
              <div className="w-8 text-right text-[11px] text-zinc-500">
                {c.confidence.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------

function DevicesCard({ world }: { world: WorldSnapshot | null }) {
  const devices = world?.devices ?? {};
  const entries = Object.entries(devices);
  return (
    <Card title="Devices" accent={`${entries.length} plug${entries.length === 1 ? "" : "s"}`}>
      {entries.length === 0 ? (
        <div className="text-sm text-zinc-500">
          No plugs discovered yet
        </div>
      ) : (
        <div className="space-y-2">
          {entries.map(([alias, d]) => (
            <div
              key={alias}
              className="flex items-center justify-between rounded border border-zinc-800 bg-zinc-950/60 px-2 py-1.5 text-sm"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${
                    d.on ? "bg-emerald-400" : "bg-zinc-600"
                  }`}
                />
                <span className="capitalize text-zinc-200">{alias}</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-xs text-zinc-400">
                  {d.power_w.toFixed(1)}W
                </span>
                <span
                  className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${
                    d.on
                      ? "bg-emerald-700 text-emerald-100"
                      : "bg-zinc-700 text-zinc-400"
                  }`}
                >
                  {d.on ? "ON" : "OFF"}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------

function BaselinesCard({
  world,
  config,
}: {
  world: WorldSnapshot | null;
  config: DashboardConfig | null;
}) {
  const b = world?.baselines;
  const calibrated = b?.calibrated ?? false;
  return (
    <Card
      title="Baselines"
      accent={
        calibrated
          ? "calibrated"
          : config
          ? `calibrating (${config.calibration_seconds}s)`
          : "unknown"
      }
    >
      {!b ? (
        <div className="text-sm text-zinc-500">Waiting…</div>
      ) : (
        <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
          <span className="text-zinc-500">Noise floor</span>
          <span className="text-right font-mono text-zinc-300">
            {b.audio_mean_db.toFixed(1)} ± {b.audio_std_db.toFixed(1)} dB
          </span>
          <span className="text-zinc-500">Typical occ.</span>
          <span className="text-right font-mono text-zinc-300">
            {b.typical_occupancy}
          </span>
          <span className="text-zinc-500">Lamp idle</span>
          <span className="text-right font-mono text-zinc-300">
            {b.power_idle_lamp_w.toFixed(1)} W
          </span>
          <span className="text-zinc-500">Fan idle</span>
          <span className="text-right font-mono text-zinc-300">
            {b.power_idle_fan_w.toFixed(1)} W
          </span>
          {b.ambient_audio_classes.length > 0 && (
            <>
              <span className="col-span-2 mt-1 text-zinc-500">Ambient</span>
              <span className="col-span-2 text-right text-[11px] text-zinc-400">
                {b.ambient_audio_classes.join(", ")}
              </span>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
