"""
Smoke test: audio capture + YAMNet classification, live in the terminal.

Run:
    python -m tests.smoke_audio

What you'll see:
    * Live dB meter bar showing current mic level.
    * Top-3 classified non-speech sounds from YAMNet.
    * Speech status (active/inactive) tracking transitions.
    * Event log for unusual_sound_class, audio_spike, speech_start/end.

How to exercise every event type:
    speech_start          — start talking after a few seconds of silence.
    speech_end            — stop talking and wait ~1s for persistence to expire.
    unusual_sound_class   — clap, knock, or snap near the mic (hold for ~1s).
    audio_spike           — clap sharply (sudden dB jump above baseline).

Keys:
    Ctrl+C   quit and print summary
"""

from perception.audio import _preview_main

if __name__ == "__main__":
    _preview_main()
