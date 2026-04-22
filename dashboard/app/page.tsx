"use client";

import VideoPanel from "@/components/VideoPanel";
import StatsPanel from "@/components/StatsPanel";
import PerceptionLog from "@/components/PerceptionLog";
import ObserverLog from "@/components/ObserverLog";
import ReasonerLog from "@/components/ReasonerLog";
import { useDashboardStream } from "@/lib/useDashboardStream";

export default function Home() {
  const {
    connection,
    config,
    world,
    events,
    observerNarrations,
    reasonerNarrations,
    routings,
  } = useDashboardStream();

  const connected = connection === "open";

  return (
    <main className="flex h-screen flex-col gap-3 overflow-hidden bg-black p-3 pb-12 text-zinc-200">
      <header className="flex shrink-0 items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">
            Newton-for-a-Room
          </h1>
          <div className="text-[11px] text-zinc-500">
            Physical AI Room Agent dashboard
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span
            className={`h-2 w-2 rounded-full ${
              connected ? "bg-emerald-400" : "bg-zinc-600"
            }`}
          />
          <span className="text-zinc-400">
            {connected
              ? "connected"
              : connection === "connecting"
              ? "connecting…"
              : "disconnected"}
          </span>
        </div>
      </header>

      {/* One content row fills the remaining viewport height. Left: the
          video + a bottom strip of room details. Right: the full-height
          pipeline column (spine + three equal panels). */}
      <section className="grid min-h-0 flex-1 grid-cols-1 gap-3 lg:grid-cols-[1fr_440px]">
        {/* LEFT — video on top, room details strip underneath. The
            stats wrapper is content-sized (shrink-0 only, no fixed
            height). That way the video flex-1 absorbs all remaining
            space and the right column's Reasoner panel bottom aligns
            with the natural bottom of the room-detail cards. */}
        <div className="flex min-h-0 flex-col gap-3">
          <div className="min-h-0 flex-1">
            <VideoPanel config={config} world={world} connected={connected} />
          </div>
          <div className="shrink-0">
            <StatsPanel world={world} config={config} />
          </div>
        </div>

        {/* RIGHT — the dataflow pipeline stack. No bottom padding so
            the Reasoner panel aligns with the bottom of the StatsPanel
            on the left. Flex ratios 4:7:7 shrink Perception to 2/3 of
            an equal third, sharing the freed space between the two
            agent panels. */}
        <div className="flex min-h-0 min-w-0 flex-col gap-3 overflow-hidden">
          <PerceptionLog
            events={events}
            className="min-h-0 flex-[4_1_0%]"
          />
          <ObserverLog
            narrations={observerNarrations}
            model={config?.agents.observer_model}
            className="min-h-0 flex-[7_1_0%]"
          />
          <ReasonerLog
            narrations={reasonerNarrations}
            routings={routings}
            model="Claude Sonnet 4.6"
            className="min-h-0 flex-[7_1_0%]"
          />
        </div>
      </section>
    </main>
  );
}
