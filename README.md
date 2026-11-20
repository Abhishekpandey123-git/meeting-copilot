# Meeting Co-Pilot & Automated Scribe

A real-time desktop HUD built with PyQt6, `faster-whisper`, and Groq (`llama-3.3-70b-versatile`) for live spoken English phrasing feedback and structured open-source meeting minutes.

## Quick Setup (Windows / PowerShell)

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
Copy-Item .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Find your audio device IDs (Microphone + VB-Cable / Stereo Mix)
python list_devices.py
# Add MIC_DEVICE_ID and SYSTEM_DEVICE_ID to .env

# 5. Launch HUD
python meeting_copilot_dual.py
```

## Notes Output
Clicking **End & Save Notes** generates a timestamped Markdown report inside the `notes/` directory containing:
- Executive Summary
- Key Technical Decisions
- Action Items (`TODO`s)
- Open Blockers & Full Transcript
