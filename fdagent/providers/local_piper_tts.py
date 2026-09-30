"""Local, offline TTS with Piper (no API, no quota, no cost). Default benchmark TTS backend.

Same interface as GeminiTTS (``synthesize`` -> PcmAudio, ``stream`` -> PCM16 chunks), so the
speech sink and kernel are unchanged. Piper yields one audio chunk per sentence; synthesis runs
in a worker thread (never on the event loop) and stops early if the consumer goes away
(barge-in). Output: mono 16-bit PCM at the voice's sample rate (22,050 Hz for the default
medium-quality voices).

Voice files (.onnx + .onnx.json) come from the rhasspy/piper-voices collection; fetch them with
``scripts/fetch_piper_voice.sh``. Configure with ``FDAGENT_PIPER_VOICE`` (path to the .onnx).
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from .tts_base import PcmAudio, TTSError, TTSStreamNotStarted

DEFAULT_PIPER_VOICE = str(Path.home() / ".cache" / "fdagent" / "piper" / "en" / "en_US" / "lessac" / "medium"
                          / "en_US-lessac-medium.onnx")


def default_voice_path() -> str:
    return os.getenv("FDAGENT_PIPER_VOICE", DEFAULT_PIPER_VOICE)


def load_piper_voice(model_path: str) -> Any:
    if not Path(model_path).exists() or not Path(model_path + ".json").exists():
        raise TTSError(f"Piper voice not found at {model_path} (+ .json); run scripts/fetch_piper_voice.sh")
    from piper import PiperVoice  # heavy import only when actually loading

    return PiperVoice.load(model_path)


class PiperTTS:
    def __init__(self, voice: Any = None, model_path: str | None = None,
                 loader: Callable[[str], Any] | None = None) -> None:
        self.model_path = model_path or default_voice_path()
        self._voice = voice
        self._loader = loader or load_piper_voice
        self._lock = threading.Lock()

    def load(self) -> Any:
        with self._lock:
            if self._voice is None:
                self._voice = self._loader(self.model_path)
            return self._voice

    @property
    def stream_sample_rate(self) -> int:
        return int(self.load().config.sample_rate)

    def _chunks(self, text: str, stop: threading.Event | None = None):
        for chunk in self.load().synthesize(text):
            if stop is not None and stop.is_set():
                return
            data = chunk.audio_int16_bytes
            if chunk.sample_channels != 1 or chunk.sample_width != 2:
                raise TTSError(f"unsupported Piper format: {chunk.sample_channels} ch x {chunk.sample_width} B")
            if data:
                yield data

    async def synthesize(self, text: str) -> PcmAudio:
        try:
            data = await asyncio.to_thread(lambda: b"".join(self._chunks(text)))
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(f"Piper synthesis failed: {type(e).__name__}: {e}") from None
        if not data:
            raise TTSError("Piper produced no audio")
        return PcmAudio(data, self.stream_sample_rate, 1)

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield sentence-sized PCM16 chunks as Piper produces them. A failure before the first
        chunk raises TTSStreamNotStarted; after it, TTSError (never restarted)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        done = object()

        def deliver(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # event loop already closed (session shut down)
                stop.set()

        def produce() -> None:
            try:
                for data in self._chunks(text, stop):
                    deliver(data)
            except Exception as e:  # delivered to the consumer in order
                deliver(e)
            finally:
                deliver(done)

        loop.run_in_executor(None, produce)
        started = False
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, Exception):
                    msg = f"Piper synthesis failed: {type(item).__name__}: {item}"
                    raise (TTSError(msg) if started else TTSStreamNotStarted(msg)) from None
                started = True
                yield item
            if not started:
                raise TTSStreamNotStarted("Piper produced no audio")
        finally:
            stop.set()  # barge-in / consumer closed: stop synthesizing further sentences

    async def aclose(self) -> None:
        pass
