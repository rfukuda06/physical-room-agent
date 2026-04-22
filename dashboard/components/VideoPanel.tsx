// VideoPanel — live MJPEG from the FastAPI backend plus client-side
// SVG overlay of the zone polygons.
//
// Two important tricks here:
//
// 1. The browser renders MJPEG (multipart JPEG) in a plain <img src=...>
//    tag with zero JavaScript. No codec, no <video> element, no Media
//    Source Extensions. The catch: once the stream dies you have to
//    re-set the src to reconnect; we handle that with a key counter.
//
// 2. The zone polygons come from /config in *camera pixel coordinates*
//    (up to 1280 x 720). Instead of computing scale factors, we put them
//    in an SVG whose viewBox matches the camera resolution. The SVG
//    scales to fit its container (w-full h-full), so its internal
//    coordinate system stays in camera pixels. Whatever the video
//    element ends up sized at, the overlay matches pixel-for-pixel.

"use client";

import { useState, useEffect } from "react";
import type { DashboardConfig, WorldSnapshot } from "@/lib/api";
import { VIDEO_STREAM_URL } from "@/lib/api";

type Props = {
  config: DashboardConfig | null;
  world: WorldSnapshot | null;
  connected: boolean;
};

export default function VideoPanel({ config, world, connected }: Props) {
  // Bump this number to force the <img> to reconnect to the stream.
  const [streamKey, setStreamKey] = useState(0);
  const [imgFailed, setImgFailed] = useState(false);

  // When the backend WebSocket reports open after being closed, it's a
  // good signal that the HTTP video stream is probably alive too — try
  // reconnecting the <img>. (The HTTP stream itself doesn't raise events
  // once it connects, so we piggyback on the WS signal.)
  useEffect(() => {
    if (connected) {
      setImgFailed(false);
      setStreamKey((k) => k + 1);
    }
  }, [connected]);

  const cameraW = config?.camera.width ?? 1280;
  const cameraH = config?.camera.height ?? 720;

  return (
    <div className="relative flex h-full min-h-0 flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900">
      {/* Fill whatever height the parent gives us. `object-contain` on
          the <img> + preserveAspectRatio="xMidYMid meet" on the SVG
          letterbox identically, so the zone overlay stays pixel-aligned
          to the video regardless of container shape. */}
      <div className="relative min-h-0 flex-1 bg-black">
        {imgFailed ? (
          <div className="flex h-full w-full items-center justify-center text-zinc-400">
            <div className="text-center">
              <div className="text-lg font-medium">Camera stream unavailable</div>
              <div className="mt-2 text-sm text-zinc-500">
                Check that <code>main.py</code> is running and the MJPEG
                endpoint is reachable.
              </div>
              <button
                type="button"
                onClick={() => {
                  setImgFailed(false);
                  setStreamKey((k) => k + 1);
                }}
                className="mt-4 rounded border border-zinc-700 px-3 py-1 text-sm hover:bg-zinc-800"
              >
                Retry
              </button>
            </div>
          </div>
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={streamKey}
            src={`${VIDEO_STREAM_URL}?k=${streamKey}`}
            alt="Live annotated camera feed"
            onError={() => setImgFailed(true)}
            className="absolute inset-0 h-full w-full object-contain"
          />
        )}

        {/* Zone polygon overlay. viewBox is the camera pixel space; the
            SVG itself scales to the container. */}
        {config && Object.keys(config.zones).length > 0 && (
          <svg
            className="pointer-events-none absolute inset-0 h-full w-full"
            viewBox={`0 0 ${cameraW} ${cameraH}`}
            preserveAspectRatio="xMidYMid meet"
            aria-hidden="true"
          >
            {Object.entries(config.zones).map(([name, pts]) => {
              const pointsStr = pts.map(([x, y]) => `${x},${y}`).join(" ");
              const labelX = pts[0]?.[0] ?? 0;
              const labelY = (pts[0]?.[1] ?? 0) - 6;
              return (
                <g key={name}>
                  <polygon
                    points={pointsStr}
                    fill="rgba(34, 197, 94, 0.08)"
                    stroke="rgba(34, 197, 94, 0.9)"
                    strokeWidth={2}
                  />
                  <text
                    x={labelX}
                    y={labelY}
                    fill="rgba(134, 239, 172, 1)"
                    fontSize={18}
                    fontFamily="ui-sans-serif, system-ui"
                    fontWeight={600}
                    paintOrder="stroke"
                    stroke="rgba(0,0,0,0.7)"
                    strokeWidth={3}
                  >
                    {name}
                  </text>
                </g>
              );
            })}
          </svg>
        )}

        {/* Corner badges: connection + people + audio */}
        <div className="pointer-events-none absolute left-3 top-3 flex gap-2">
          <StatusBadge
            label={connected ? "LIVE" : "OFFLINE"}
            color={connected ? "bg-red-500" : "bg-zinc-600"}
            pulse={connected}
          />
          {world && (
            <StatusBadge
              label={`${world.people_count} ${
                world.people_count === 1 ? "person" : "people"
              }`}
              color="bg-zinc-800"
            />
          )}
        </div>

        {world?.audio && (
          <div className="pointer-events-none absolute right-3 top-3">
            <StatusBadge
              label={`${world.audio.level_db.toFixed(0)} dB${
                world.audio.speech_active ? " • SPEECH" : ""
              }`}
              color={
                world.audio.recent_spike
                  ? "bg-amber-600"
                  : world.audio.speech_active
                  ? "bg-sky-700"
                  : "bg-zinc-800"
              }
            />
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({
  label,
  color,
  pulse,
}: {
  label: string;
  color: string;
  pulse?: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-white ${color}`}
    >
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-70" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-white" />
        </span>
      )}
      {label}
    </span>
  );
}
