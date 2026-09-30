"""Whisper transcription core (no LiveKit imports), used inside the dedicated STT process.

Kept free of LiveKit/asyncio so the STT child process imports only numpy + transformers.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np

# base.en: on the local synthetic regression set (numbers, IDs, dates, names; clean and Opus
# round-trip) it heard 9-10/10 key values vs 8-9/10 for tiny.en, fixing spoken-ID errors, at ~2.7 s
# for a 6 s turn with a 4-thread cap on this CPU (tiny.en ~1.5 s). Override: FDAGENT_WHISPER_MODEL.
DEFAULT_WHISPER_MODEL = "openai/whisper-base.en"
TARGET_RATE = 16000

PipelineFactory = Callable[[str, str], Any]


def _transformers_pipeline(model_id: str, device: str) -> Any:
    from transformers import pipeline  # heavy import, only when actually loading

    return pipeline("automatic-speech-recognition", model=model_id, device=device)


def max_new_tokens_for(seconds: float) -> int:
    """Generous cap on decoded tokens for a segment (speech is ~3-4 tokens/s). Prevents a
    runaway/hallucinating decode from occupying the STT process for many seconds."""
    return int(min(440, 24 + 8 * seconds))


class WhisperTranscriber:
    """Thread-safe, lazily loaded Whisper. ``transcribe`` takes float32 mono 16 kHz audio."""

    def __init__(self, model_id: str = DEFAULT_WHISPER_MODEL, device: str = "cpu",
                 pipeline_factory: PipelineFactory | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._factory = pipeline_factory or _transformers_pipeline
        self._pipe: Any = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        with self._lock:
            if self._pipe is None:
                self._pipe = self._factory(self.model_id, self.device)
            return self._pipe

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        pipe = self.load()
        gen: dict[str, Any] = {"max_new_tokens": max_new_tokens_for(audio.size / TARGET_RATE)}
        if not self.model_id.endswith(".en"):
            gen.update({"language": "english", "task": "transcribe"})
        with self._lock:  # one inference at a time per model instance
            out = pipe({"raw": audio, "sampling_rate": TARGET_RATE}, generate_kwargs=gen)
        text = out.get("text", "") if isinstance(out, dict) else str(out)
        return text.strip()


_CACHE: dict[tuple[str, str], WhisperTranscriber] = {}


def get_transcriber(model_id: str = DEFAULT_WHISPER_MODEL, device: str = "cpu") -> WhisperTranscriber:
    """Process-wide cache: load the model once per process."""
    key = (model_id, device)
    if key not in _CACHE:
        _CACHE[key] = WhisperTranscriber(model_id, device)
    return _CACHE[key]
