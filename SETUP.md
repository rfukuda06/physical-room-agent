# SETUP.md

Quick checklist to get the Newton-for-a-Room dev environment running at the start of a session.

---

## 1. Activate the Python virtual environment

The repo already has a `venv/` directory at the project root (Python 3.11). You need to activate it every time you open a new terminal — it isolates this project's packages from the rest of your system.

```bash
cd ~/Desktop/Repos/archetype_mvp
source venv/bin/activate
```

You'll know it worked when your shell prompt gets a `(venv)` prefix. To leave it later: `deactivate`.

> **Why this matters:** without activating, `python` and `pip` point at your system Python, so installed packages (YOLO, FastAPI, etc.) won't be found. If you ever see `ModuleNotFoundError: No module named 'ultralytics'` — you forgot to activate.

## 2. Confirm dependencies are installed

First time on a new machine, or after pulling changes that touched `requirements.txt`:

```bash
pip install -r requirements.txt
```

Sanity check that the big ones imported cleanly:

```bash
python -c "import ultralytics, tensorflow, anthropic, google.generativeai, fastapi; print('ok')"
```

## 3. Confirm `.env` exists and is populated

The project loads secrets from `.env` at the repo root via `python-dotenv` (see `config.py:14`). `.env` is gitignored — never commit it.

Required keys:

```
GEMINI_API_KEY=...
CLAUDE_API_KEY=...
KASA_USERNAME=...       # email you use for the Kasa app
KASA_PASSWORD=...       # Kasa account password (needed for KLAP auth on KP125M)
```

If any are missing, `config.py` will raise `KeyError` on import. Test with:

```bash
python -c "import config; print('config loaded')"
```

## 4. Check hardware is connected

Before running `main.py` for a real session:

- **Camera** — built-in or USB. Indexed by `CAMERA_INDEX` in `config.py` (default `0`).
- **Microphone** — indexed by `AUDIO_DEVICE_INDEX` (default `0`, MacBook Air mic). To list devices:
  ```bash
  python -c "import sounddevice as sd; print(sd.query_devices())"
  ```
- **Kasa smart plugs** — both the lamp and fan plugs need to be on the same Wi-Fi/hotspot network as your laptop. IPs can change after a hotspot restart; we re-discover at startup via the plug *alias* (`light`, `fan`). Quick check:
  ```bash
  kasa --username "$KASA_USERNAME" --password "$KASA_PASSWORD" discover
  ```

## 4b. Camera placement (read before Block 4 onward)

The agent's "where is the person?" logic assumes the camera sees the **floor**
clearly. This matters because zones (defined in Block 4) are polygons drawn
on the floor, and a person's zone is decided by where their feet project
into the image. If the camera is at face height — the laptop default — feet
vanish behind desks and the zone labels lie. This is worth getting right
*once* instead of fighting it every session.

### Recommended hardware

- **Best:** external USB webcam on a tripod or shelf mount (e.g. Logitech
  C270/C920 class, ~$20-40). Any 720p+ webcam is fine — YOLO doesn't need 4K.
- **Workable:** prop the laptop on a high shelf, lid open, screen facing
  into the room. You'll need a second device (phone, another laptop) to
  actually watch the dashboard.

### Placement checklist

1. **Height 1.8–2.2 m.** Top of a bookshelf, mounted on a wall, or over a
   doorframe. Higher than standing head height keeps feet visible when
   someone walks toward the camera.
2. **Tilt ~15–25° downward.** Enough to see the floor around each zone;
   not so steep that you lose upper-body pose cues.
3. **Corner of the room, pointing across the diagonal.** Maximizes coverage
   of desk + door + couch in a single frame with the fewest occlusions.
4. **A standing person at the far wall fills ~50–70% of frame height.** If
   they're tiny, the camera is too far; if they're cropped, it's too close.
5. **No windows in direct view.** Backlight defeats auto-exposure —
   people become silhouettes and YOLO misses them.
6. **Stable mount.** Every time the camera shifts, your zones drift and you
   re-run the zone tool. Gaffer tape or a tripod; not a stack of books.

### What to avoid

- **Camera at desk level** (the default laptop position). Person sits down
  → bbox collapses into the desk → feet are occluded → zone queries lie.
- **Straight-down ceiling view.** Sounds ideal; isn't. YOLO-pose was trained
  on mostly-upright views, so keypoints degrade badly, and you lose the
  sit-vs-stand signal entirely.
- **Moving the camera between sessions** without re-running the zone tool.
  Pixel zones are tied to *this specific* camera pose.

---

## 4c. Mapping zones with the click tool

Once the camera is placed, populate `config.ZONES` with polygons that name
the floor regions of your room (desk, door, couch, etc.). You do this once
per camera pose via the interactive click tool.

### When to (re)run it

- After initial setup.
- Any time the camera physically moves.
- If you rearrange furniture enough that the desk / couch / door shifted
  within the frame.

If none of the above happened, your existing `config.ZONES` is still valid —
don't re-run just because.

### Walkthrough

1. **Start the tool.**
   ```bash
   source venv/bin/activate
   python -m perception.zone_map
   ```
   A window titled `zone_map (q=quit)` opens with the live camera feed. Any
   zones already in `config.ZONES` are pre-drawn so you can see prior work.

2. **Think floor, not object.** For each zone, trace the *floor area* around
   the furniture — where a person standing at / using the object would have
   their feet. Not the surface of the desk itself. Typical zones are 4–6
   sided polygons on the floor immediately in front of / around each piece
   of furniture.

3. **For each zone you want to define:**
   - In the **terminal**, type a short name (`desk`, `door`, `couch`) and
     press Enter.
   - In the **window**, left-click the corners in order (clockwise or
     counter-clockwise, either works). Each click drops a red dot; lines
     connect them live.
   - Keys while drawing:
     - `u` — undo the last point.
     - `r` — reset this zone and start its corners over.
     - `Enter` — finish the polygon (needs at least 3 points). The zone
       locks in and is redrawn in a distinct color.
     - `q` — quit the tool entirely.
   - After `Enter`, the terminal prints a ready-to-paste line:
     ```
     "desk": [(412, 603), (887, 598), (901, 842), (388, 847)],
     ```

4. **Finish.** Type an empty zone name (just press Enter at the prompt) or
   press `q` in the window. The tool prints one final block with all
   captured zones:
   ```python
   ZONES: dict[str, list[tuple[int, int]]] = {
       "desk":  [(412, 603), (887, 598), (901, 842), (388, 847)],
       "door":  [(1055, 420), (1260, 418), (1258, 880), (1050, 885)],
       "couch": [(120, 700), (610, 695), (605, 940), (115, 945)],
   }
   ```

5. **Paste into `config.py`.** Replace the existing `ZONES = { ... }` block
   (around lines 74–84) with the printed block. Save.

6. **Verify.** Either re-run the tool (existing zones will be pre-drawn so
   you can eyeball their correctness), or run the smoke test:
   ```bash
   python -m tests.smoke_zone_map
   ```
   Walk into and out of each zone; watch the red foot-dot's label flip
   between zone names and `unknown`.

### Intuition for good zones

- **Bigger is better than smaller.** You want a zone to trigger reliably
  when someone is "at" that spot, not only when they stand on one exact
  tile. Err on the generous side — extending a meter past the furniture
  is usually fine.
- **Overlap is allowed.** The floor between the desk and door is
  legitimately part of both. `zone_for_entity` returns a *list*; downstream
  code decides what to do with ambiguity.
- **Keep zones off the far wall.** Polygons near the top of the frame (i.e.
  near the image vanishing line) suffer from perspective: tiny pixel errors
  translate to huge floor-position errors. Put meaningful zones in the
  lower two-thirds of the frame.

---

## 5. Run the orchestrator

```bash
python main.py
```

Right now this only prints the loaded config — the full pipeline (perception → agents → actuators) is still being wired up.

## 6. Running the dashboard (later, not yet built)

When the FastAPI server + Next.js dashboard exist, they'll be started separately:

```bash
# Terminal A — backend
uvicorn server.main:app --reload

# Terminal B — frontend
cd dashboard && npm run dev
```

---

## Common gotchas

- **`source venv/bin/activate` not activating** — make sure you're in the repo root, not `~/Desktop/Repos/`.
- **`KeyError: 'GEMINI_API_KEY'`** — `.env` is missing or not at the repo root. `config.py` uses `Path(__file__).parent / ".env"`, so it must sit next to `config.py`.
- **TensorFlow / Keras version warnings at import** — expected; we pin `tf_keras` alongside `tensorflow` for YAMNet compatibility.
- **Kasa plugs not discovered** — phone hotspot probably rotated IPs or the plugs dropped off Wi-Fi. Power-cycle the plugs or re-run `kasa discover`.
- **"RuntimeError: PortAudio not found"** on macOS — `brew install portaudio`, then reinstall `sounddevice`.

## Deactivating when done

```bash
deactivate
```

This just drops the `venv`'s shims from your PATH — nothing is deleted, and next session you just `source venv/bin/activate` again.
