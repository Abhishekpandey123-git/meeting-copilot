import sys
import os
import queue
import threading
import datetime
import json
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame
)
from faster_whisper import WhisperModel
from groq import Groq

load_dotenv()

def _parse_device_id(val: str | None):
    if val is None or val.strip() == "":
        return None
    try:
        return int(val.strip())
    except ValueError:
        return val.strip()

# -----------------------------
# Configuration
# -----------------------------
MIC_DEVICE_ID = _parse_device_id(os.getenv("MIC_DEVICE_ID"))
SYSTEM_DEVICE_ID = _parse_device_id(os.getenv("SYSTEM_DEVICE_ID"))
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 1.0
MIN_SPEECH_DURATION = 0.8

os.makedirs("notes", exist_ok=True)
NOTES_FILE = os.path.join(
    "notes",
    f"meeting_notes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
)

# -----------------------------
# Event Bus
# -----------------------------
class EventBus(QObject):
    transcript_ready = pyqtSignal(str)
    suggestion_ready = pyqtSignal(str)
    status_updated = pyqtSignal(str)

bus = EventBus()

# -----------------------------
# Dual-Stream Speech & Coaching Worker
# -----------------------------
class DualSpeechWorker(threading.Thread):
    def __init__(self, audio_queue: queue.Queue):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.running = True
        self.transcript_log = []
        self.log_lock = threading.Lock()

        # Uses CUDA float16 if available (e.g., RTX 2050), falls back to CPU int8
        try:
            self.whisper = WhisperModel(WHISPER_MODEL_NAME, device="cuda", compute_type="float16")
            self.device_label = "GPU (CUDA)"
        except Exception:
            self.whisper = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
            self.device_label = "CPU (INT8)"

        self.groq_client = Groq()

    def run(self):
        bus.status_updated.emit(f"Listening • {self.device_label}")
        while self.running:
            item = self.audio_queue.get()
            if item is None:
                break

            speaker, audio_data = item
            segments, _ = self.whisper.transcribe(
                audio_data,
                beam_size=1,
                language="en",
                condition_on_previous_text=False
            )
            text = " ".join([s.text.strip() for s in segments]).strip()
            if not text:
                continue

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            with self.log_lock:
                self.transcript_log.append(f"[{timestamp}] {speaker}: {text}")

            if speaker == "You":
                bus.transcript_ready.emit(f"You ({timestamp}): {text}")
                threading.Thread(target=self.evaluate_english, args=(text,), daemon=True).start()
            else:
                bus.transcript_ready.emit(f"Team ({timestamp}): {text}")

    def evaluate_english(self, user_text: str):
        prompt = f"""
You are a real-time spoken English co-pilot for an open-source software engineer in a live technical meeting.
User Spoken: "{user_text}"

Evaluate grammar, prepositions, and natural open-source engineering phrasing (PRs, issues, CI, architecture).
If natural and clear, return: {{"action": "pass"}}
If awkward or grammatically off, return:
{{
  "action": "suggest",
  "better": "<Sharp, natural alternative under 15 words>"
}}
Output strictly valid JSON.
"""
        try:
            resp = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(resp.choices[0].message.content)
            if result.get("action") == "suggest" and result.get("better"):
                bus.suggestion_ready.emit(result["better"])
        except Exception as e:
            print(f"[Groq Coaching Error]: {e}")

    def generate_meeting_summary(self):
        with self.log_lock:
            if not self.transcript_log:
                print("[Notes] No speech captured.")
                return
            raw_transcript = "\n".join(self.transcript_log)

        summary_prompt = f"""
You are a technical scribe for an open-source engineering weekly sync.
Analyze the dual-channel transcript below (`You` vs `Team`):
---
{raw_transcript}
---

Generate structured Markdown meeting notes containing:
## 1. Executive Summary
## 2. Key Technical Decisions
## 3. Action Items (`TODO`s with owner `[You]` or `[Team]`)
## 4. Open Blockers & Follow-ups
"""
        try:
            resp = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.2
            )
            with open(NOTES_FILE, "w", encoding="utf-8") as f:
                f.write(resp.choices[0].message.content)
                f.write("\n\n---\n### Raw Transcript\n```text\n" + raw_transcript + "\n```\n")
            print(f"[Notes] Saved to {os.path.abspath(NOTES_FILE)}")
        except Exception as e:
            print(f"[Notes Export Error]: {e}")

# -----------------------------
# VAD Audio Stream Wrapper
# -----------------------------
class VADStream:
    def __init__(self, speaker_label: str, device_id, output_queue: queue.Queue):
        self.speaker_label = speaker_label
        self.device_id = device_id
        self.output_queue = output_queue
        self.buffer = []
        self.silence_chunks = 0
        self.chunk_size = int(SAMPLE_RATE * 0.1)
        self.silence_limit = int(SILENCE_DURATION / 0.1)
        self.min_chunks = int(MIN_SPEECH_DURATION / 0.1)
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        frame = indata[:, 0]
        rms = float(np.sqrt(np.mean(frame ** 2)))

        if rms > SILENCE_THRESHOLD:
            self.buffer.append(frame.copy())
            self.silence_chunks = 0
        elif self.buffer:
            self.buffer.append(frame.copy())
            self.silence_chunks += 1
            if self.silence_chunks >= self.silence_limit:
                if len(self.buffer) >= self.min_chunks:
                    segment = np.concatenate(self.buffer, axis=0)
                    self.output_queue.put((self.speaker_label, segment))
                self.buffer = []
                self.silence_chunks = 0

    def start(self):
        self.stream = sd.InputStream(
            device=self.device_id,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=self.chunk_size,
            dtype="float32",
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

# -----------------------------
# Floating HUD Overlay
# -----------------------------
class MeetingHUD(QWidget):
    def __init__(self, worker: DualSpeechWorker):
        super().__init__()
        self.worker = worker
        self.drag_position = None
        self.init_ui()

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SubWindow
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 23, 42, 0.94);
                border: 1px solid #334155;
                border-radius: 10px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.status_label = QLabel("INITIALIZING...")
        self.status_label.setStyleSheet("color: #64748b; font-size: 11px; font-weight: bold;")
        header.addWidget(self.status_label)
        header.addStretch()

        self.btn_export = QPushButton("End & Save Notes")
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                padding: 4px 12px;
                font-size: 11px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1d4ed8; }
        """)
        self.btn_export.clicked.connect(self.export_and_quit)
        header.addWidget(self.btn_export)
        layout.addLayout(header)

        self.transcript_label = QLabel("Ready. Speak or listen to start live transcription...")
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(self.transcript_label)

        self.suggestion_label = QLabel("💡 Phrasing upgrades will appear here.")
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.setStyleSheet("color: #38bdf8; font-size: 14px; font-weight: 600;")
        layout.addWidget(self.suggestion_label)

        root.addWidget(card)

        screen = QApplication.primaryScreen().geometry()
        width, height = 740, 115
        self.setGeometry((screen.width() - width) // 2, 20, width, height)

        bus.transcript_ready.connect(self.transcript_label.setText)
        bus.suggestion_ready.connect(lambda s: self.suggestion_label.setText(f"💡 Say instead: {s}"))
        bus.status_updated.connect(lambda st: self.status_label.setText(st.upper()))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position:
            self.move(event.globalPosition().toPoint() - self.drag_position)

    def export_and_quit(self):
        self.btn_export.setEnabled(False)
        self.status_label.setText("GENERATING MEETING NOTES...")
        threading.Thread(target=self._finish, daemon=True).start()

    def _finish(self):
        self.worker.generate_meeting_summary()
        QApplication.quit()

# -----------------------------
# Entry Point
# -----------------------------
def main():
    if not os.getenv("GROQ_API_KEY"):
        print("[Error] Set GROQ_API_KEY in .env or environment first.")
        sys.exit(1)

    app = QApplication(sys.argv)
    audio_q = queue.Queue()

    worker = DualSpeechWorker(audio_q)
    worker.start()

    mic_stream = VADStream("You", MIC_DEVICE_ID, audio_q)
    mic_stream.start()

    sys_stream = None
    if SYSTEM_DEVICE_ID is not None:
        sys_stream = VADStream("Team", SYSTEM_DEVICE_ID, audio_q)
        sys_stream.start()

    hud = MeetingHUD(worker)
    hud.show()

    code = app.exec()

    mic_stream.stop()
    if sys_stream:
        sys_stream.stop()
    audio_q.put(None)
    worker.join(timeout=2.0)
    sys.exit(code)

if __name__ == "__main__":
    main()
