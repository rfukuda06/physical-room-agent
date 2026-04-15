"""
Zone map — name the floor regions of the room.

Given a point (x, y) in camera-pixel space, answer: *which named zones contain
it?* Given a YoloEntity, answer the same question using the best query point
for that entity (ankle-keypoint midpoint when confident, bbox-bottom fallback).

--- The core geometric idea (read this once) ---

A single webcam projects the 3D room onto a 2D image, so pixel coordinates
can't directly tell us where a person is in the room — EXCEPT under one
crucial assumption: *the person's feet touch the floor*. If that's true, the
ray from the camera through the foot pixel intersects the floor plane at
exactly one point. So a person's foot pixel *does* uniquely identify their
floor location, no formal calibration required.

That's why zones in this project are drawn on the FLOOR, not around objects.
A "desk" zone is the region of floor in front of the desk where someone
standing at the desk has their feet — not the desk surface itself. The
click-to-define tool at the bottom of this file is used to trace those floor
polygons by eye on a live camera frame.

--- Why ankle keypoints, with a bbox-bottom fallback ---

For a standing person fully in frame, the midpoint of the two ankle keypoints
is the closest thing we have to "where the feet are on the floor." YOLO-pose
gives us 17 COCO keypoints per person; indices 15 and 16 are left/right
ankles. Each keypoint carries a confidence.

When a person sits behind a desk, the desk physically occludes their ankles,
so those keypoints either get low confidence or drift onto the desk surface.
In that case we fall back to the bottom-center of the bounding box. This
isn't floor truth — it's "the lowest visible pixel of the person" — but it's
robust and, for zones drawn generously around furniture, usually still lands
inside the right polygon.

--- Why multi-zone membership is a feature ---

`zones_for_point` returns a *list* of zones, not a single zone. If the user
draws overlapping polygons (floor between desk and door), a point can
legitimately be in both. Downstream code (event_detector.py) decides whether
a transition counts or is noise. Forcing mutual exclusion would bake a
judgment call into the wrong layer.

--- Running the click-to-define tool ---

    python -m perception.zone_map

See SETUP.md §4c for the full walkthrough. Short version: type a zone name,
left-click its corners on the frame, press Enter to close the polygon, paste
the printed block into config.ZONES.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from perception.yolo_engine import YoloEntity

# COCO-pose keypoint indices. YOLO-pose models return 17 keypoints per person
# in this order; we only need the ankles.
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16

# Minimum per-keypoint confidence to trust an ankle. If either ankle falls
# below this, we fall back to bbox-bottom. 0.5 is the same threshold the
# Ultralytics plot() helper uses when deciding whether to render a keypoint.
ANKLE_CONF_MIN = 0.5


# ---------------------------------------------------------------------------
# Zone cache — loaded lazily from config.ZONES on first query.
# ---------------------------------------------------------------------------

# Cached list of (name, polygon-as-int32-ndarray). OpenCV's pointPolygonTest
# wants shape (N, 1, 2) int32; we normalize once at load time so every query
# is a cheap C call, not a per-call reshape.
_zones_cache: Optional[list[tuple[str, np.ndarray]]] = None
_cache_lock = threading.Lock()


def _load_zones() -> list[tuple[str, np.ndarray]]:
    """Read config.ZONES, validate, and return the int32 polygon cache.

    Zones with fewer than 3 vertices are skipped with a warning — a
    "polygon" of 2 points is a line segment and cv2.pointPolygonTest
    would reject it anyway.
    """
    import config

    out: list[tuple[str, np.ndarray]] = []
    for name, pts in config.ZONES.items():
        if len(pts) < 3:
            print(f"[zone_map] skipping '{name}': polygon needs >= 3 points, got {len(pts)}")
            continue
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        out.append((name, poly))
    return out


def _ensure_loaded() -> list[tuple[str, np.ndarray]]:
    global _zones_cache
    with _cache_lock:
        if _zones_cache is None:
            _zones_cache = _load_zones()
        return _zones_cache


def reload_zones() -> None:
    """Force a re-read of config.ZONES on next query.

    Useful after editing config.py in a long-running process (e.g. the
    click tool writes zones, you paste them, and want the smoke test to
    pick them up without restarting).
    """
    global _zones_cache
    with _cache_lock:
        _zones_cache = None


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

def zones_for_point(x: float, y: float) -> list[str]:
    """Return every zone whose polygon contains (x, y). Empty list if none.

    Uses cv2.pointPolygonTest with measureDist=False: returns +1 for inside,
    0 for exactly on the edge, -1 for outside. We treat edge-hits as inside.
    """
    out: list[str] = []
    for name, poly in _ensure_loaded():
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
            out.append(name)
    return out


def _foot_point(entity: YoloEntity) -> tuple[float, float]:
    """Pick the best "where is this entity on the floor" pixel for an entity.

    Person + confident ankles  -> ankle midpoint (truest to the floor).
    Person + occluded ankles    -> bbox bottom-center (lowest visible pixel).
    Non-person                  -> bbox center (objects don't have feet, and
                                    they usually don't straddle zones).
    """
    cx, cy, w, h = entity.bbox_xywh

    if entity.cls_name != "person":
        return cx, cy

    # Pose fields only exist when yolo_engine saw a pose-model result.
    if (
        entity.keypoints_xy is not None
        and entity.keypoints_conf is not None
        and len(entity.keypoints_xy) > RIGHT_ANKLE_IDX
    ):
        lxy = entity.keypoints_xy[LEFT_ANKLE_IDX]
        rxy = entity.keypoints_xy[RIGHT_ANKLE_IDX]
        lc = entity.keypoints_conf[LEFT_ANKLE_IDX]
        rc = entity.keypoints_conf[RIGHT_ANKLE_IDX]
        if lc >= ANKLE_CONF_MIN and rc >= ANKLE_CONF_MIN:
            return (lxy[0] + rxy[0]) * 0.5, (lxy[1] + rxy[1]) * 0.5

    # Fallback: bottom-center of the bounding box.
    return cx, cy + h * 0.5


def zone_for_entity(entity: YoloEntity) -> list[str]:
    """Return the list of zones containing this entity's foot point."""
    fx, fy = _foot_point(entity)
    return zones_for_point(fx, fy)


# ---------------------------------------------------------------------------
# Click-to-define tool: python -m perception.zone_map
#
# Opens the live camera feed and lets the user trace polygon zones by
# clicking their corners. Prints ready-to-paste config entries.
# ---------------------------------------------------------------------------

# Distinct BGR colors we cycle through for finalized zones (helps you see
# what you've already captured while drawing a new one).
_ZONE_COLORS = [
    (0, 255, 0),     # green
    (255, 128, 0),   # blue-ish
    (0, 128, 255),   # orange
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (128, 0, 255),   # pink
]

# Color for the polygon currently being drawn.
_ACTIVE_COLOR = (0, 0, 255)    # red


def _draw_dashed_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    dash_len: int = 10,
    gap_len: int = 6,
    thickness: int = 2,
) -> None:
    """Draw a dashed straight line from p1 to p2 onto img in place.

    OpenCV doesn't ship a dashed-line primitive, so we walk the segment in
    dash_len+gap_len steps and draw short solid pieces. Used to preview the
    polygon-closing edge while the user is still clicking.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length < 1.0:
        return
    step = dash_len + gap_len
    n = int(length // step) + 1
    ux, uy = dx / length, dy / length
    for i in range(n):
        start_d = i * step
        end_d = min(start_d + dash_len, length)
        if start_d >= length:
            break
        sx = int(x1 + ux * start_d)
        sy = int(y1 + uy * start_d)
        ex = int(x1 + ux * end_d)
        ey = int(y1 + uy * end_d)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


def _overlay_zones(
    frame: np.ndarray,
    finalized: list[tuple[str, list[tuple[int, int]]]],
    active_name: Optional[str],
    active_points: list[tuple[int, int]],
) -> np.ndarray:
    """Draw finalized zones + the polygon-in-progress onto a copy of `frame`."""
    out = frame.copy()

    # Finalized zones — filled outline + label.
    for i, (name, pts) in enumerate(finalized):
        color = _ZONE_COLORS[i % len(_ZONE_COLORS)]
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2)
        # Translucent fill so stacked zones stay visible.
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
        cv2.putText(
            out, name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )

    # Polygon in progress — dots + connecting lines in red.
    if active_points:
        # Solid red edges between clicked points.
        if len(active_points) >= 2:
            poly = np.array(active_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                out, [poly], isClosed=False, color=_ACTIVE_COLOR, thickness=2
            )
        # Preview the closing edge (last -> first) once the polygon would be
        # valid on Enter. Drawn as a dashed line so it's visually distinct
        # from real edges: this is "what the shape will be if you finalize
        # now," not a clicked edge.
        if len(active_points) >= 3:
            _draw_dashed_line(out, active_points[-1], active_points[0], _ACTIVE_COLOR)
        # Draw dots on top of lines so corners are obvious.
        for i, pt in enumerate(active_points):
            # First point gets a hollow ring so "close-the-loop" target is clear.
            cv2.circle(out, pt, 7, _ACTIVE_COLOR, 2 if i == 0 else -1)
        if active_name:
            hint = f"drawing: {active_name} ({len(active_points)} pts)"
            if len(active_points) >= 3:
                hint += "  — ENTER to close"
            cv2.putText(
                out, hint, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _ACTIVE_COLOR, 2, cv2.LINE_AA,
            )

    # Hint strip along the bottom.
    cv2.putText(
        out,
        "click=add pt  ENTER=finish zone  u=undo  r=reset  q=quit",
        (10, out.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def _click_tool_main() -> None:
    """Interactive zone-capture tool. See module docstring / SETUP.md §4c."""
    import config
    from perception.camera import CameraCapture

    camera = CameraCapture(
        camera_index=config.CAMERA_INDEX,
        capture_size=(config.CAMERA_CAPTURE_WIDTH, config.CAMERA_CAPTURE_HEIGHT),
        capture_fps=config.CAMERA_CAPTURE_FPS,
        buffer_size=(config.BUFFER_FRAME_WIDTH, config.BUFFER_FRAME_HEIGHT),
        buffer_fps=config.BUFFER_FPS,
        buffer_seconds=config.FRAME_BUFFER_SECONDS,
    )

    print("zone_map click tool — trace FLOOR polygons, not objects.")
    print("See SETUP.md §4c for a full walkthrough.")
    camera.start()

    # Wait for first frame so we don't show an empty window.
    for _ in range(100):
        if camera.latest_frame() is not None:
            break
        time.sleep(0.05)

    # Seed finalized zones with whatever's already in config — lets the user
    # see prior work and add/overwrite without starting from scratch.
    finalized: list[tuple[str, list[tuple[int, int]]]] = [
        (name, list(pts)) for name, pts in config.ZONES.items()
    ]
    if finalized:
        print(f"Loaded {len(finalized)} existing zone(s) from config.ZONES: "
              f"{[n for n, _ in finalized]}")

    # Mouse state shared between the OpenCV callback and the main loop.
    active_points: list[tuple[int, int]] = []
    polygon_closed = threading.Event()   # set by Enter key in the window

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            active_points.append((int(x), int(y)))

    win = "zone_map (q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    quit_requested = False
    try:
        while not quit_requested:
            # ---- Prompt for the next zone name in the terminal ----
            try:
                zone_name = input("\nZone name (empty to finish): ").strip()
            except EOFError:
                zone_name = ""
            if not zone_name:
                break

            active_points.clear()
            polygon_closed.clear()
            print(
                f"  drawing '{zone_name}': click corners in the window. "
                "ENTER to finish, u=undo, r=reset, q=quit."
            )

            # ---- Inner render/key loop for this one polygon ----
            while True:
                frame = camera.latest_frame()
                if frame is not None:
                    disp = _overlay_zones(frame, finalized, zone_name, active_points)
                    cv2.imshow(win, disp)

                key = cv2.waitKey(20) & 0xFF
                if key == ord("q"):
                    quit_requested = True
                    break
                if key == ord("u"):
                    if active_points:
                        active_points.pop()
                if key == ord("r"):
                    active_points.clear()
                # Enter = 13 (CR) on macOS; accept 10 (LF) too just in case.
                if key in (10, 13):
                    if len(active_points) < 3:
                        print(f"  need at least 3 points, got {len(active_points)}. keep clicking.")
                        continue
                    break

            if quit_requested:
                break

            # Lock in the zone. Overwrite if a zone with this name already exists.
            pts_snapshot = list(active_points)
            finalized = [(n, p) for n, p in finalized if n != zone_name]
            finalized.append((zone_name, pts_snapshot))

            line = f'    "{zone_name}": {pts_snapshot},'
            print(f"  captured '{zone_name}' ({len(pts_snapshot)} pts)")
            print("  paste-ready line:")
            print(line)

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    # ---- Final copy-pasteable block ----
    if finalized:
        print("\n" + "=" * 60)
        print("Paste this into config.py (replace the existing ZONES = { ... }):")
        print("=" * 60)
        print("ZONES: dict[str, list[tuple[int, int]]] = {")
        for name, pts in finalized:
            print(f'    "{name}": {pts},')
        print("}")
    else:
        print("\nNo zones captured.")


if __name__ == "__main__":
    _click_tool_main()
