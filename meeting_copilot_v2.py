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
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QLineEdit
)
from faster_whisper import WhisperModel
from groq import Groq
from vad_engine import SpeechSegmenter

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

os.makedirs("notes", exist_ok=True)
os.makedirs("sessions", exist_ok=True)
SESSION_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
NOTES_FILE = os.path.join("notes", f"meeting_notes_{SESSION_STAMP}.md")
JSONL_FILE = os.path.join("sessions", f"transcript_{SESSION_STAMP}.jsonl")

# -----------------------------
# Event Bus
# -----------------------------
class EventBus(QObject):
    transcript_ready = pyqtSignal(str)
    suggestion_ready = pyqtSignal(str)
    draft_ready = pyqtSignal(str)
    status_updated = pyqtSignal(str)

bus = EventBus()

# -----------------------------
# Dual-Stream AI Engine
# -----------------------------
class CopilotEngine(threading.Thread):
    def __init__(self, audio_queue: queue.Queue):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.running = True
        self.paused = False
        self.transcript_log = []
        self.log_lock = threading.Lock()

        try:
            self.whisper = WhisperModel(WHISPER_MODEL_NAME, device="cuda", compute_type="float16")
            self.device_label = "CUDA FP16"
        except Exception:
            self.whisper = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
            self.device_label = "CPU INT8"

        self.groq = Groq()

    def run(self):
        bus.status_updated.emit(f"LIVE • {self.device_label}")
        while self.running:
            item = self.audio_queue.get()
            if item is None:
                break
            if self.paused:
                continue

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

            ts = datetime.datetime.now().strftime("%H:%M:%S")
            entry = {"time": ts, "speaker": speaker, "text": text}

            with self.log_lock:
                self.transcript_log.append(f"[{ts}] {speaker}: {text}")
                with open(JSONL_FILE, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(entry, ensure_ascii=False) + "\n")

            bus.transcript_ready.emit(f"{speaker} ({ts}): {text}")

            if speaker == "You":
                threading.Thread(target=self.evaluate_english, args=(text,), daemon=True).start()

    def _recent_context(self, n: int = 4) -> str:
        with self.log_lock:
            return "\n".join(self.transcript_log[-n:])

    def evaluate_english(self, user_text: str):
        context = self._recent_context(4)
        prompt = f"""
You are a real-time spoken English co-pilot for an open-source engineer in a weekly sync.
Recent meeting context:
{context}

User just said: "{user_text}"

If the phrasing is clear, natural, and idiomatic for an engineering call, return: {{"action": "pass"}}
If there is a grammar flaw, awkward phrasing, or a sharper technical way to say it, return:
{{
  "action": "suggest",
  "better": "<Natural, concise alternative under 16 words>"
}}
Output strictly valid JSON.
"""
        try:
            resp = self.groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(resp.choices[0].message.content)
            if result.get("action") == "suggest" and result.get("better"):
                bus.suggestion_ready.emit(result["better"])
        except Exception as e:
            print(f"[Groq Coach Error]: {e}")

    def phrase_from_notes(self, rough_idea: str):
        context = self._recent_context(5)
        prompt = f"""
You are a live speaking assistant for an open-source developer in a meeting.
Recent discussion:
{context}

Convert the developer's rough bullet/idea into ONE crisp, natural spoken sentence they can read aloud right now:
Rough idea: "{rough_idea}"

Return ONLY the spoken sentence, no quotes or commentary.
"""
        try:
            resp = self.groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            bus.draft_ready.emit(resp.choices[0].message.content.strip())
        except Exception as e:
            bus.draft_ready.emit(f"Error: {e}")

    def generate_meeting_summary(self) -> str | None:
        with self.log_lock:
            if not self.transcript_log:
                return None
            raw_transcript = "\n".join(self.transcript_log)

        summary_prompt = f"""
You are a technical scribe for an open-source engineering weekly meeting.
Analyze this dual-channel transcript (`You` vs `Team`):
---
{raw_transcript}
---

Write clean Markdown meeting notes with these sections:
## 1. Executive Summary
## 2. Key Technical Decisions
## 3. Action Items (`TODO`s with owner `[You]` or `[Team]` and referenced PRs/issues)
## 4. Open Blockers & Follow-ups
"""
        try:
            resp = self.groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.2
            )
            with open(NOTES_FILE, "w", encoding="utf-8") as f:
                f.write(resp.choices[0].message.content)
                f.write("\n\n---\n### Raw Transcript\n```text\n" + raw_transcript + "\n```\n")
            return os.path.abspath(NOTES_FILE)
        except Exception as e:
            print(f"[Notes Error]: {e}")
            return None

# -----------------------------
# Audio Stream Capture (512-sample Silero frames)
# -----------------------------
class DualAudioStream:
    def __init__(self, speaker_label: str, device_id, output_queue: queue.Queue):
        self.speaker_label = speaker_label
        self.device_id = device_id
        self.output_queue = output_queue
        self.segmenter = SpeechSegmenter(sample_rate=SAMPLE_RATE)
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        frame = indata[:, 0]
        segment = self.segmenter.process_frame(frame)
        if segment is not None:
            self.output_queue.put((self.speaker_label, segment))

    def start(self):
        self.stream = sd.InputStream(
            device=self.device_id,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=512,
            dtype="float32",
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

# -----------------------------
# Floating HUD Overlay (v2)
# -----------------------------
class MeetingHUDv2(QWidget):
    def __init__(self, engine: CopilotEngine):
        super().__init__()
        self.engine = engine
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
                background-color: rgba(15, 23, 42, 0.95);
                border: 1px solid #334155;
                border-radius: 10px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(7)

        # Top Bar
        header = QHBoxLayout()
        self.status_label = QLabel("INITIALIZING...")
        self.status_label.setStyleSheet("color: #34d399; font-size: 11px; font-weight: bold;")
        header.addWidget(self.status_label)
        header.addStretch()

        self.btn_mute = QPushButton("Pause")
        self.btn_mute.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #e2e8f0;
                border: none;
                padding: 4px 10px;
                font-size: 11px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        self.btn_mute.clicked.connect(self.toggle_pause)
        header.addWidget(self.btn_mute)

        self.btn_checkpoint = QPushButton("Save Checkpoint")
        self.btn_checkpoint.setStyleSheet("""
            QPushButton {
                background-color: #0f766e;
                color: white;
                border: none;
                padding: 4px 10px;
                font-size: 11px;
                border-radius: 4px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #115e59; }
        """)
        self.btn_checkpoint.clicked.connect(self.save_checkpoint)
        header.addWidget(self.btn_checkpoint)

        self.btn_export = QPushButton("End & Save")
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

        # Live Transcript
        self.transcript_label = QLabel("Listening for You & Team...")
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setStyleSheet("color: #94a3b8; font-size: 12.5px;")
        layout.addWidget(self.transcript_label)

        # Real-time Suggestion / Say-This Box
        self.suggestion_label = QLabel("💡 Phrasing feedback & drafted sentences appear here.")
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.setStyleSheet("color: #38bdf8; font-size: 14px; font-weight: 600;")
        layout.addWidget(self.suggestion_label)

        # Quick Prompt Input ("What do I say?")
        prompt_row = QHBoxLayout()
        self.idea_input = QLineEdit()
        self.idea_input.setPlaceholderText("Type rough thought (e.g. 'ci failing on windows, fixing tonight') -> Enter to phrase...")
        self.idea_input.setStyleSheet("""
            QLineEdit {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 5px;
                padding: 5px 8px;
                font-size: 12px;
            }
            QLineEdit:focus { border: 1px solid #38bdf8; }
        """)
        self.idea_input.returnPressed.connect(self.trigger_phrase_assist)
        prompt_row.addWidget(self.idea_input)
        layout.addLayout(prompt_row)

        root.addWidget(card)

        screen = QApplication.primaryScreen().geometry()
        width, height = 780, 155
        self.setGeometry((screen.width() - width) // 2, 18, width, height)

        # Signals
        bus.transcript_ready.connect(self.transcript_label.setText)
        bus.suggestion_ready.connect(lambda s: self.suggestion_label.setText(f"💡 Say instead: {s}"))
        bus.draft_ready.connect(lambda d: self.suggestion_label.setText(f"🗣️ Read aloud: {d}"))
        bus.status_updated.connect(lambda st: self.status_label.setText(st.upper()))

        # Hotkey Ctrl+Space to focus the quick-phrase bar
        QShortcut(QKeySequence("Ctrl+Space"), self, activated=self.idea_input.setFocus)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position:
            self.move(event.globalPosition().toPoint() - self.drag_position)

    def toggle_pause(self):
        self.engine.paused = not self.engine.paused
        if self.engine.paused:
            self.btn_mute.setText("Resume")
            self.status_label.setText("PAUSED")
            self.status_label.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold;")
        else:
            self.btn_mute.setText("Pause")
            self.status_label.setText(f"LIVE • {self.engine.device_label}")
            self.status_label.setStyleSheet("color: #34d399; font-size: 11px; font-weight: bold;")

    def trigger_phrase_assist(self):
        idea = self.idea_input.text().strip()
        if not idea:
            return
        self.idea_input.clear()
        self.suggestion_label.setText("⏳ Drafting natural phrasing...")
        threading.Thread(target=self.engine.phrase_from_notes, args=(idea,), daemon=True).start()

    def save_checkpoint(self):
        self.status_label.setText("SAVING CHECKPOINT...")
        threading.Thread(target=self._checkpoint_worker, daemon=True).start()

    def _checkpoint_worker(self):
        path = self.engine.generate_meeting_summary()
        if path:
            bus.status_updated.emit(f"CHECKPOINT SAVED • {self.engine.device_label}")
        else:
            bus.status_updated.emit(f"NO SPEECH YET • {self.engine.device_label}")

    def export_and_quit(self):
        self.btn_export.setEnabled(False)
        self.status_label.setText("GENERATING FINAL NOTES...")
        threading.Thread(target=self._finish, daemon=True).start()

    def _finish(self):
        self.engine.generate_meeting_summary()
        QApplication.quit()

# -----------------------------
# Entry Point
# -----------------------------
def main():
    if not os.getenv("GROQ_API_KEY"):
        print("[Error] Set GROQ_API_KEY in .env first.")
        sys.exit(1)

    app = QApplication(sys.argv)
    audio_q = queue.Queue()

    engine = CopilotEngine(audio_q)
    engine.start()

    mic_stream = DualAudioStream("You", MIC_DEVICE_ID, audio_q)
    mic_stream.start()

    sys_stream = None
    if SYSTEM_DEVICE_ID is not None:
        sys_stream = DualAudioStream("Team", SYSTEM_DEVICE_ID, audio_q)
        sys_stream.start()

    hud = MeetingHUDv2(engine)
    hud.show()

    code = app.exec()

    mic_stream.stop()
    if sys_stream:
        sys_stream.stop()
    audio_q.put(None)
    engine.join(timeout=2.0)
    sys.exit(code)

if __name__ == "__main__":
    main()
