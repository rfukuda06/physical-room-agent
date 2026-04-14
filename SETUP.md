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
