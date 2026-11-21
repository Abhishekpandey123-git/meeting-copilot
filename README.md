# Meeting Co-Pilot v2

Upgraded dual-stream spoken English co-pilot and automated open-source meeting scribe.

## What's New in v2
1. **Silero VAD (`vad_engine.py`)**: Neural voice activity detection replaces basic RMS volume thresholds—ignores keyboard typing and fan noise.
2. **Context-Aware Coaching**: Evaluates your spoken English against the last 4 turns of the meeting so phrasing suggestions match what your teammates just asked.
3. **Live "What Do I Say?" Input Bar (`Ctrl+Space`)**: Type a rough technical bullet (e.g., `pr ready after rebasing main`) and hit `Enter` to get an instant, natural sentence to read aloud.
4. **Crash-Safe JSONL Logging & Mid-Call Checkpoints**: Streams every turn to `sessions/transcript_*.jsonl` in real time and lets you export Markdown notes mid-meeting via **Save Checkpoint**.
5. **Pause / Resume Toggle**: Mute transcription and coaching anytime during off-topic breaks.

## Setup & Run (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python list_devices.py
# Add GROQ_API_KEY, MIC_DEVICE_ID, and SYSTEM_DEVICE_ID to .env
python meeting_copilot_v2.py
```
