import numpy as np
import torch

class SpeechSegmenter:
    def __init__(self, sample_rate: int = 16000, silence_duration: float = 0.85, min_speech: float = 0.7):
        self.sample_rate = sample_rate
        self.chunk_samples = 512  # 32ms at 16kHz (required by Silero VAD v5)
        self.silence_limit = int((silence_duration * sample_rate) / self.chunk_samples)
        self.min_chunks = int((min_speech * sample_rate) / self.chunk_samples)

        self.buffer = []
        self.silence_chunks = 0
        self.speaking = False

        try:
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
            self.model.eval()
            self.use_silero = True
        except Exception as e:
            print(f"[VAD] Silero fallback to RMS energy ({e})")
            self.model = None
            self.use_silero = False

    def _is_speech(self, chunk: np.ndarray) -> bool:
        if self.use_silero and self.model is not None:
            tensor = torch.from_numpy(chunk).float()
            with torch.no_grad():
                prob = self.model(tensor, self.sample_rate).item()
            return prob > 0.55
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        return rms > 0.015

    def process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        is_voice = self._is_speech(frame)

        if is_voice:
            self.speaking = True
            self.buffer.append(frame.copy())
            self.silence_chunks = 0
        elif self.speaking:
            self.buffer.append(frame.copy())
            self.silence_chunks += 1
            if self.silence_chunks >= self.silence_limit:
                segment = None
                if len(self.buffer) >= self.min_chunks:
                    segment = np.concatenate(self.buffer, axis=0)
                self.buffer = []
                self.silence_chunks = 0
                self.speaking = False
                return segment
        return None
