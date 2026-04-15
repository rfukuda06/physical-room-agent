"""
YOLO (Ultralytics) — fast structured perception.

Runs detection, tracking (BoT-SORT), and pose estimation on each frame the
camera hands us. Emits structured machine state — entities with IDs, bboxes,
and 17 pose keypoints per person. Does NOT narrate or interpret; pose-class
labels (sitting / standing / walking) are derived later in event_detector.

--- Quick primer on the concepts in here (for first-time readers) ---

* `ultralytics.YOLO("yolo26n-pose.pt")` — the Ultralytics wrapper class. It
  auto-downloads the weights file from their GitHub releases on first use,
  loads the PyTorch model, and gives us a `.track()` / `.predict()` API.
  YOLO26 is their latest model family (Jan 2026 release); the `n` size is
  "nano" (~2.4M params) which is the right choice for real-time CPU/MPS
  inference on a laptop.

* `.track()` vs `.predict()` — `predict` runs only detection. `track` wraps
  `predict` AND runs a tracker on the output so every box carries a stable
  `track_id` across frames. Passing `persist=True` keeps the tracker's
  internal state between calls (without it, IDs reset every frame and the
  whole thing is useless). BoT-SORT is the default tracker; it combines
  motion prediction with appearance features to re-identify people.

* Pose keypoints — for each person, YOLO returns 17 (x, y, confidence)
  triplets following the COCO skeleton: nose, eyes, ears, shoulders, elbows,
  wrists, hips, knees, ankles. That's what later layers use to classify
  sitting vs. standing.

* Threading mirrors `camera.py`. Inference is 20-40ms per frame on Apple
  Silicon MPS, so we run it in its own daemon thread. Consumers call
  `engine.latest_result()` without blocking on a forward pass.

* Apple Silicon GPU ("mps") — PyTorch calls Apple's Metal backend "mps"
  (Metal Performance Shaders). It's ~3x faster than CPU for this model.
  On non-Mac machines, or if MPS fails, we fall back to "cpu" at startup.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Result types — plain dataclasses so later layers can JSON-serialize trivially.
# ---------------------------------------------------------------------------

@dataclass
class YoloEntity:
    """One detected object in a single frame."""
    track_id: Optional[int]                          # None if tracker didn't assign one
    cls_name: str                                     # e.g. "person"
    cls_id: int
    bbox_xywh: tuple[float, float, float, float]     # (center_x, center_y, w, h) in pixels
    conf: float
    # Pose fields are populated only when the model is a pose variant AND
    # the entity is a person. Each list is length-17 (COCO skeleton).
    keypoints_xy: Optional[list[tuple[float, float]]] = None
    keypoints_conf: Optional[list[float]] = None


@dataclass
class YoloResult:
    """A full YOLO pass on one camera frame."""
    timestamp: float                      # time.monotonic() when inference finished
    frame_shape: tuple[int, int]          # (H, W) of the source frame
    entities: list[YoloEntity] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None  # BGR image with boxes + skeleton drawn
    infer_ms: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class YoloEngine:
    """Background YOLO worker.

    Pulls frames from a CameraCapture, runs detection + tracking + pose,
    and publishes the most recent YoloResult behind a lock. Consumers
    poll with `latest_result()`.
    """

    def __init__(
        self,
        camera,                              # perception.camera.CameraCapture
        model_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        tracker: str,
        device: str,
        infer_every_n_frames: int = 1,
    ) -> None:
        self._camera = camera
        self._model_path = model_path
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._tracker = tracker
        self._device = device
        self._infer_every = max(1, infer_every_n_frames)

        self._model = None           # loaded in start()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._result_lock = threading.Lock()
        self._latest: Optional[YoloResult] = None

        # Stats — protected by the same lock.
        self._frames_processed = 0
        self._infer_ms_sum = 0.0

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("YoloEngine already started")

        # Imported lazily so `import perception` doesn't pay a multi-second
        # torch import tax on every module load.
        from ultralytics import YOLO

        self._model = YOLO(self._model_path)

        # First inference on a dummy frame warms the device and catches
        # device-specific errors at startup rather than mid-stream.
        try:
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self._model.predict(
                dummy, imgsz=self._imgsz, device=self._device, verbose=False
            )
        except Exception as e:
            print(f"[yolo] device={self._device!r} failed on warmup ({e}); falling back to cpu")
            self._device = "cpu"
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self._model.predict(dummy, imgsz=self._imgsz, device="cpu", verbose=False)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="yolo-engine", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> "YoloEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ---- consumer API ----

    def latest_result(self) -> Optional[YoloResult]:
        """Return the most recent YoloResult (annotated frame copied)."""
        with self._result_lock:
            r = self._latest
            if r is None:
                return None
            annotated = r.annotated_frame.copy() if r.annotated_frame is not None else None
            # entities/bbox tuples are immutable value types — safe to share.
            return YoloResult(
                timestamp=r.timestamp,
                frame_shape=r.frame_shape,
                entities=list(r.entities),
                annotated_frame=annotated,
                infer_ms=r.infer_ms,
            )

    def stats(self) -> dict:
        with self._result_lock:
            n = self._frames_processed
            avg = (self._infer_ms_sum / n) if n else 0.0
        return {"frames_processed": n, "avg_infer_ms": round(avg, 1)}

    # ---- internals ----

    def _run_loop(self) -> None:
        """Background thread: read latest camera frame, run YOLO, publish."""
        assert self._model is not None
        last_frame_id: int = 0
        frame_ix = 0

        while not self._stop_event.is_set():
            frame = self._camera.latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Skip if the camera hasn't produced a NEW frame since last time.
            # Using id() is fine: CameraCapture hands us a fresh copy each call,
            # so a new object == a new frame. Cheap and avoids redundant work.
            fid = id(frame)
            if fid == last_frame_id:
                time.sleep(0.005)
                continue
            last_frame_id = fid

            frame_ix += 1
            if frame_ix % self._infer_every != 0:
                continue

            t0 = time.monotonic()
            try:
                # `persist=True` keeps BoT-SORT's track memory across calls —
                # without it, every call starts fresh and IDs would churn.
                results = self._model.track(
                    frame,
                    imgsz=self._imgsz,
                    conf=self._conf,
                    iou=self._iou,
                    tracker=self._tracker,
                    device=self._device,
                    persist=True,
                    verbose=False,
                )
            except Exception as e:
                print(f"[yolo] inference error: {e}")
                time.sleep(0.05)
                continue
            infer_ms = (time.monotonic() - t0) * 1000.0

            r0 = results[0]
            entities = _extract_entities(r0)
            annotated = r0.plot()  # draws boxes, track IDs, and pose skeleton

            result = YoloResult(
                timestamp=time.monotonic(),
                frame_shape=frame.shape[:2],
                entities=entities,
                annotated_frame=annotated,
                infer_ms=infer_ms,
            )

            with self._result_lock:
                self._latest = result
                self._frames_processed += 1
                self._infer_ms_sum += infer_ms


def _extract_entities(ultra_result) -> list[YoloEntity]:
    """Convert an Ultralytics `Results` object into plain YoloEntity dataclasses.

    We pull numpy arrays out of the GPU/MPS tensors once, up-front, rather
    than touching `.boxes.xywh` (which triggers a device->host copy) inside
    a loop. The tensors have shape [N, ...] where N is the detection count.
    """
    boxes = ultra_result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = ultra_result.names  # {class_id: class_name}
    xywh = boxes.xywh.cpu().numpy()                   # (N, 4)
    confs = boxes.conf.cpu().numpy()                  # (N,)
    cls_ids = boxes.cls.cpu().numpy().astype(int)     # (N,)
    track_ids = (
        boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    )

    # Pose keypoints — present only on -pose models.
    kps_xy = None
    kps_conf = None
    if ultra_result.keypoints is not None and ultra_result.keypoints.xy is not None:
        kps_xy = ultra_result.keypoints.xy.cpu().numpy()        # (N, 17, 2)
        if ultra_result.keypoints.conf is not None:
            kps_conf = ultra_result.keypoints.conf.cpu().numpy()  # (N, 17)

    entities: list[YoloEntity] = []
    for i in range(len(boxes)):
        kp_xy_list = None
        kp_conf_list = None
        if kps_xy is not None:
            kp_xy_list = [(float(x), float(y)) for x, y in kps_xy[i]]
            if kps_conf is not None:
                kp_conf_list = [float(c) for c in kps_conf[i]]

        entities.append(
            YoloEntity(
                track_id=int(track_ids[i]) if track_ids is not None else None,
                cls_name=names.get(int(cls_ids[i]), str(cls_ids[i])),
                cls_id=int(cls_ids[i]),
                bbox_xywh=(
                    float(xywh[i, 0]),
                    float(xywh[i, 1]),
                    float(xywh[i, 2]),
                    float(xywh[i, 3]),
                ),
                conf=float(confs[i]),
                keypoints_xy=kp_xy_list,
                keypoints_conf=kp_conf_list,
            )
        )
    return entities


# ---------------------------------------------------------------------------
# Standalone preview: `python -m perception.yolo_engine`
#
# Starts the camera + engine and shows an annotated window with boxes, IDs,
# and pose skeleton. Logs one summary line per second to the console.
#   q -> quit (prints final stats)
# ---------------------------------------------------------------------------

def _preview_main() -> None:
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

    print("Starting camera + YOLO preview — press 'q' to quit.")
    camera.start()
    # Wait for the first camera frame so the engine warmup sees real data too.
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
                cv2.imshow("yolo (q=quit)", result.annotated_frame)

            # One log line per second.
            now = time.monotonic()
            if result is not None and now - last_log >= 1.0:
                ids = sorted({e.track_id for e in result.entities if e.track_id is not None})
                persons = sum(1 for e in result.entities if e.cls_name == "person")
                others = len(result.entities) - persons
                print(
                    f"[yolo] {result.infer_ms:5.1f}ms | persons={persons} other={others} "
                    f"ids={ids}"
                )
                last_log = now

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        stats = engine.stats()
        engine.stop()
        camera.stop()
        cv2.destroyAllWindows()
        print(f"YOLO stats on exit: {stats}")


if __name__ == "__main__":
    _preview_main()
