// Central constants + TypeScript types mirroring the FastAPI message
// contract in server/broadcaster.py. Keep in sync when shapes change.

// HTTP endpoints go through the Next.js rewrite proxy (see next.config.ts)
// so the browser sees them as same-origin. This is what finally makes the
// MJPEG <img> stream render — cross-origin multipart/x-mixed-replace is
// unreliable in browsers even with permissive CORS. Relative paths work
// because the browser resolves them against whatever host served the page.
export const VIDEO_STREAM_URL = "/backend/video/stream";
export const CONFIG_URL = "/backend/config";

// WebSocket can't use the Next.js rewrite proxy (rewrites are HTTP-only),
// so we connect directly to the backend. CORS doesn't apply to WebSocket
// handshakes the same way, so cross-origin WS works fine. The port must
// match main.py's run_server_in_thread; override via NEXT_PUBLIC_BACKEND_PORT
// in .env.local so a port change in one place doesn't require code edits.
const BACKEND_PORT = process.env.NEXT_PUBLIC_BACKEND_PORT || "8005";
export const STATE_WS_URL =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:${BACKEND_PORT}/ws/state`
    : `ws://127.0.0.1:${BACKEND_PORT}/ws/state`;

// ---- Config (one-shot on page load) ----
export type DashboardConfig = {
  camera: { width: number; height: number; fps: number };
  zones: Record<string, [number, number][]>;
  thresholds: { audio_spike_db: number; yamnet_min_conf: number };
  agents: {
    observer_enabled: boolean;
    observer_model: string;
    reasoner_enabled: boolean;
  };
  calibration_seconds: number;
};

// ---- Snapshot (WorldState, pushed ~10 Hz) ----
export type Entity = {
  id: number;
  bbox_xywh: [number, number, number, number];
  zones: string[];
  pose: "standing" | "sitting" | "walking" | "unknown";
  seconds_in_frame: number;
  velocity_px_per_s: number;
};

export type AudioSnapshot = {
  level_db: number;
  top_classes: { label: string; confidence: number }[];
  dominant_class: string;
  speech_active: boolean;
  recent_spike: boolean;
  spike_magnitude_db: number;
};

export type DeviceSnapshot = { on: boolean; power_w: number };

export type Baselines = {
  audio_mean_db: number;
  audio_std_db: number;
  typical_occupancy: number;
  power_idle_lamp_w: number;
  power_idle_fan_w: number;
  ambient_audio_classes: string[];
  calibrated: boolean;
};

export type WorldSnapshot = {
  timestamp: string;
  entities: Entity[];
  people_count: number;
  audio: AudioSnapshot;
  devices: Record<string, DeviceSnapshot>;
  baselines: Baselines;
  scene_description?: string;
  activity_summary?: string;
  recent_events?: EventMsg[];
};

// ---- Event (fired on each Layer 0 event) ----
export type EventMsg = {
  type: string;
  ts: number;
  elapsed?: number;
  track_id: number | null;
  zones: string[];
  payload: Record<string, unknown>;
};

// ---- Narration (Observer / Reasoner output) ----
// Fields are optional so the one type fits both agents:
//   Observer sends: narration, escalate, escalate_reason, trigger_events, ...
//   Reasoner sends: narration, lamp, fan, alert, speak, reasoning, ...
export type NarrationMsg = {
  narration: string;
  ts: number;
  // Observer-only
  escalate?: boolean;
  escalate_reason?: string;
  trigger_events?: string[];
  // Reasoner-only
  lamp?: "on" | "off" | null;
  fan?: "on" | "off" | null;
  alert?: boolean;
  speak?: boolean;
  reasoning?: string;
  // Either
  world_state_update?: Record<string, unknown>;
};

// ---- Routing (Reasoner routing decision, even without Claude) ----
export type RoutingMsg = {
  trigger: string;
  fired: boolean;
  escalate: boolean;
  reason: string;
  ts: number;
};

// ---- Tagged union for WS messages ----
export type WSMessage =
  | { kind: "snapshot"; data: WorldSnapshot }
  | { kind: "event"; data: EventMsg }
  | { kind: "narration"; agent: "observer" | "reasoner"; data: NarrationMsg }
  | { kind: "routing"; data: RoutingMsg };
