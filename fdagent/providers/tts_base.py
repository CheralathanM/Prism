"""Provider-neutral TTS types shared by the Gemini and local (Piper) backends and the speech sink."""

from __future__ import annotations

from dataclasses import dataclass


class TTSError(RuntimeError):
    pass


class TTSStreamNotStarted(TTSError):
    """Streaming failed before any audio was produced; callers may fall back to unary."""


@dataclass(frozen=True)
class PcmAudio:
    data: bytes  # little-endian int16
    sample_rate: int
    channels: int = 1
