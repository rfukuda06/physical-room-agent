// Client-side hook that subscribes to the backend WebSocket and splits
// incoming messages into three React state slices.
//
// React hook mental model (for first-timers):
//   * A hook is a function that lets components "hook into" React
//     state and lifecycle. Rules: only call from the top level of a
//     component or another hook; never inside conditions/loops.
//   * useState returns [value, setter]. Calling the setter triggers
//     a re-render of the component tree that uses that state.
//   * useEffect runs *after* render. The return function is cleanup,
//     called before the next effect run and on unmount — perfect for
//     closing sockets when the component goes away.
//
// This hook connects to /ws/state once on mount and auto-reconnects
// with a 1s delay on disconnect. Every message mutates at most one of
// the three slices — consumers only re-render when their slice changes
// (React bails on identical references by default for most state).

"use client";

import { useEffect, useRef, useState } from "react";
import type {
  DashboardConfig,
  EventMsg,
  NarrationMsg,
  RoutingMsg,
  WSMessage,
  WorldSnapshot,
} from "./api";
import { CONFIG_URL, STATE_WS_URL } from "./api";

const EVENT_BUFFER = 200;       // scrollback for the event log
const NARRATION_BUFFER = 100;   // scrollback for agent logs

export type ConnectionState = "connecting" | "open" | "closed";

// Every stored entry gets tagged with a monotonically increasing localId
// (so React keys are unique even within the same millisecond) and a wall-
// clock receivedAt (for rendering). Python's event.ts is monotonic with
// an arbitrary origin, so it's not directly convertible to clock time —
// using arrival time at the browser is simpler and consistent across all
// three log streams.
type Tag = { _localId: number; _receivedAt: number };

export type DashboardStream = {
  connection: ConnectionState;
  config: DashboardConfig | null;
  world: WorldSnapshot | null;
  events: (EventMsg & Tag)[];
  observerNarrations: (NarrationMsg & Tag)[];
  reasonerNarrations: (NarrationMsg & Tag)[];
  routings: (RoutingMsg & Tag)[];
};

export function useDashboardStream(): DashboardStream {
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [config, setConfig] = useState<DashboardConfig | null>(null);
  const [world, setWorld] = useState<WorldSnapshot | null>(null);
  const [events, setEvents] = useState<(EventMsg & Tag)[]>([]);
  const [observerNarrations, setObserverNarrations] = useState<
    (NarrationMsg & Tag)[]
  >([]);
  const [reasonerNarrations, setReasonerNarrations] = useState<
    (NarrationMsg & Tag)[]
  >([]);
  const [routings, setRoutings] = useState<(RoutingMsg & Tag)[]>([]);

  // Refs hold values that persist across renders without triggering them.
  // We use one for the WS handle (for cleanup) and one for a monotonic
  // local id that tags narrations/routings so React keys stay unique even
  // when two arrive in the same millisecond.
  const wsRef = useRef<WebSocket | null>(null);
  const localIdRef = useRef<number>(0);
  const cancelledRef = useRef<boolean>(false);

  // Fetch /config once on mount. Not in the WS stream because it's static.
  useEffect(() => {
    fetch(CONFIG_URL)
      .then((r) => r.json())
      .then((data: DashboardConfig) => setConfig(data))
      .catch((err) => console.warn("config fetch failed:", err));
  }, []);

  // Open the WebSocket and wire up reconnect.
  useEffect(() => {
    cancelledRef.current = false;

    const connect = () => {
      if (cancelledRef.current) return;
      setConnection("connecting");
      const ws = new WebSocket(STATE_WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnection("open");

      ws.onmessage = (ev) => {
        let msg: WSMessage;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        const receivedAt = Date.now();
        switch (msg.kind) {
          case "snapshot":
            setWorld(msg.data);
            break;
          case "event": {
            const tagged = {
              ...msg.data,
              _localId: ++localIdRef.current,
              _receivedAt: receivedAt,
            };
            setEvents((prev) => {
              const next = [...prev, tagged];
              return next.length > EVENT_BUFFER
                ? next.slice(-EVENT_BUFFER)
                : next;
            });
            break;
          }
          case "narration": {
            const tagged = {
              ...msg.data,
              _localId: ++localIdRef.current,
              _receivedAt: receivedAt,
            };
            if (msg.agent === "observer") {
              setObserverNarrations((prev) => {
                const next = [...prev, tagged];
                return next.length > NARRATION_BUFFER
                  ? next.slice(-NARRATION_BUFFER)
                  : next;
              });
            } else {
              setReasonerNarrations((prev) => {
                const next = [...prev, tagged];
                return next.length > NARRATION_BUFFER
                  ? next.slice(-NARRATION_BUFFER)
                  : next;
              });
            }
            break;
          }
          case "routing": {
            const tagged = {
              ...msg.data,
              _localId: ++localIdRef.current,
              _receivedAt: receivedAt,
            };
            setRoutings((prev) => {
              const next = [...prev, tagged];
              return next.length > NARRATION_BUFFER
                ? next.slice(-NARRATION_BUFFER)
                : next;
            });
            break;
          }
        }
      };

      ws.onclose = () => {
        setConnection("closed");
        wsRef.current = null;
        if (!cancelledRef.current) {
          // Reconnect after 1s — keeps the dashboard alive across
          // backend restarts during development.
          setTimeout(connect, 1000);
        }
      };

      ws.onerror = () => {
        // Trigger the onclose path by closing explicitly. Browsers
        // normally do this themselves but being explicit makes the
        // reconnect timing deterministic.
        try {
          ws.close();
        } catch {
          /* no-op */
        }
      };
    };

    connect();

    return () => {
      // Cleanup on unmount: cancel any pending reconnect and drop the WS.
      cancelledRef.current = true;
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  return {
    connection,
    config,
    world,
    events,
    observerNarrations,
    reasonerNarrations,
    routings,
  };
}
