# tests/ — interactive smoke tests

These are **smoke tests**, not pytest unit tests. You run one, watch the
live camera window + console, and judge correctness by eye. They're the
Day-1-style "does the real pipeline behave right?" checks that match the
project's manual-loop-closure philosophy.

Run one at a time (they all grab the webcam):

```bash
source venv/bin/activate

python -m tests.smoke_camera       # webcam + rolling 10s buffer
python -m tests.smoke_yolo         # + YOLO boxes, track IDs, pose skeleton
python -m tests.smoke_zone_map     # + config.ZONES polygons + per-person foot points
```

Press `q` in the preview window to quit.

### When to add pytest tests here

When a module has a pure data contract that's cheap to exercise without
hardware (e.g. `agents/world_state.py` diffing), drop a `test_*.py` file in
this directory. `pytest tests/` will pick it up automatically while leaving
the `smoke_*.py` scripts alone.

### Related entrypoints

* `python -m perception.zone_map` — click-to-define tool for populating
  `config.ZONES`. Not a test; it writes the config you then paste in.
  See `SETUP.md` §4c for the walkthrough.
* `python -m perception.camera` / `python -m perception.yolo_engine` — the
  original module-level previews still work; `smoke_camera.py` and
  `smoke_yolo.py` just forward to them.
