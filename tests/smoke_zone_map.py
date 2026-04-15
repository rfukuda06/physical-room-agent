"""
Smoke test: zones + YOLO, end-to-end.

Run:
    python -m tests.smoke_zone_map

What you'll see:
    * Camera feed annotated with YOLO boxes + track IDs + pose skeleton.
    * Every zone from config.ZONES drawn as a green outline.
    * For each person detected, a red dot at their computed foot-point
      (ankle midpoint when confident, bbox-bottom otherwise) labeled with
      track_id and the list of zones containing that foot-point.
    * One console line per second summarizing each person's foot-point and
      zones.

Walk into / out of your zones; sit behind the desk; confirm the overlays and
log lines match what you'd expect. Press 'q' to quit.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import config
from perception.camera import CameraCapture
from perception.yolo_engine import YoloEngine
from perception.zone_map import _foot_point, zone_for_entity


ZONE_COLOR = (0, 255, 0)          # green
FOOT_COLOR = (0, 0, 255)          # red
LABEL_COLOR = (255, 255, 255)     # white


def _draw_overlay(frame: np.ndarray, entities) -> np.ndarray:
    """Overlay zone polygons + foot points onto the annotated YOLO frame."""
    out = frame

    # Zone polygons.
    for name, pts in config.ZONES.items():
        if len(pts) < 3:
            continue
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], isClosed=True, color=ZONE_COLOR, thickness=2)
        cv2.putText(
            out, name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            ZONE_COLOR, 2, cv2.LINE_AA,
        )

    # Per-person foot points + zone labels.
    for ent in entities:
        if ent.cls_name != "person":
            continue
        fx, fy = _foot_point(ent)
        zones = zone_for_entity(ent)
        cv2.circle(out, (int(fx), int(fy)), 8, FOOT_COLOR, -1)
        label = f"id={ent.track_id} " + (",".join(zones) if zones else "unknown")
        cv2.putText(
            out, label, (int(fx) + 10, int(fy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_COLOR, 2, cv2.LINE_AA,
        )

    return out


def main() -> None:
    camera = CameraCapture(
        camera_index=config.CAMERA_INDEX,
        capture_size=(config.CAMERA_CAPTURE_WIDTH, config.CAMERA_CAPTURE_HEIGHT),
        capture_fps=config.CAMERA_CAPTURE_FPS,
        buffer_size=(config.BUFFER_FRAME_WIDTH, config.BUFFER_FRAME_HEIGHT),
        buffer_fps=config.BUFFER_FPS,
        buffer_seconds=config.FRAME_BUFFER_SECONDS,
    )
    engine = YoloEngine(
        camera=camera,
        model_path=config.YOLO_MODEL,
        imgsz=config.YOLO_IMGSZ,
        conf=config.YOLO_CONF,
        iou=config.YOLO_IOU,
        tracker=config.YOLO_TRACKER,
        device=config.YOLO_DEVICE,
        infer_every_n_frames=config.YOLO_INFER_EVERY_N_FRAMES,
    )

    n_zones = len(config.ZONES)
    print(f"zone_map smoke test — {n_zones} zone(s) loaded: "
          f"{list(config.ZONES.keys()) or '(none — run python -m perception.zone_map first)'}")
    print("Press 'q' in the window to quit.")

    camera.start()
    for _ in range(100):
        if camera.latest_frame() is not None:
            break
        time.sleep(0.05)
    engine.start()

    last_log = 0.0
    try:
        while True:
            result = engine.latest_result()
            if result is not None and result.annotated_frame is not None:
                frame = _draw_overlay(result.annotated_frame, result.entities)
                cv2.imshow("zone_map smoke (q=quit)", frame)

                now = time.monotonic()
                if now - last_log >= 1.0:
                    for ent in result.entities:
                        if ent.cls_name != "person":
                            continue
                        fx, fy = _foot_point(ent)
                        zones = zone_for_entity(ent)
                        print(
                            f"[zones] id={ent.track_id} "
                            f"foot=({int(fx)}, {int(fy)}) zones={zones}"
                        )
                    last_log = now

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
