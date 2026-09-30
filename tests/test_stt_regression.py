"""Local STT regression set: numbers, IDs, dates, destinations, product names (synthetic only).

Audio is synthesized locally with Piper, then optionally passed through an Opus encode/decode
round trip (like LiveKit's WebRTC path) before Whisper transcribes it. Measures key-value
accuracy, word error rate and latency per Whisper checkpoint.

Opt-in (needs Piper, a Piper voice, Whisper weights and ffmpeg):
    FDAGENT_RUN_STT_REGRESSION=1 pytest tests/test_stt_regression.py -s
    python -m tests.test_stt_regression openai/whisper-tiny.en openai/whisper-base.en
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time

import pytest

# (spoken text, key values that must be recognisable; each key is a tuple of accepted forms)
PHRASES: list[tuple[str, list[tuple[str, ...]]]] = [
    ("Could you track order ZX-4471 for me?", [("zx4471",)]),
    ("Please convert seven hundred fifty euros to Japanese yen.", [("750", "seven hundred fifty"), ("euro", "€"), ("yen",)]),
    ("I want flights to Lisbon on April 7th.", [("lisbon",), ("april 7", "april seventh")]),
    ("Search for a standing lamp under eighty dollars.", [("standing lamp",), ("80", "eighty")]),
    ("Update my library card number to K 5 5 2 1 9.", [("k55219",)]),
    ("Find a three bedroom home in Northfield under two thousand one hundred a month.",
     [("northfield",), ("3", "three"), ("2100", "2 100", "two thousand one hundred")]),
    ("My trip is on the 23rd of November, actually make it the 25th.", [("november",), ("25",)]),
    ("Add two bottles of olive soap to my cart.", [("olive soap",), ("2", "two")]),
    ("How long is the bus ride from Harbor Street to Maple Avenue?", [("harbor",), ("maple",)]),
    ("Set the maximum rent filter to nineteen fifty.", [("1950", "nineteen fifty")]),
]

_NUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9"}


def norm_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower().replace(",", ""))


def compact(s: str) -> str:
    """Letters/digits only, spelled single digits folded (for IDs like 'K 5 5 2 1 9')."""
    toks = norm_words(s)
    return "".join(_NUM.get(t, t) for t in toks)


def key_hit(hyp: str, forms: tuple[str, ...]) -> bool:
    h_words, h_compact, h_raw = " ".join(norm_words(hyp)), compact(hyp), hyp.lower()
    return any((f in h_words) or (f.replace(" ", "") in h_compact) or (not f.isalnum() and f in h_raw)
               for f in forms)


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def _to_16k(pcm: bytes, rate: int, opus: bool):
    import numpy as np

    args = ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0"]
    if opus:  # WebRTC-like path: 48 kHz Opus at a modest voice bitrate, decoded back
        enc = subprocess.run(args + ["-ar", "48000", "-c:a", "libopus", "-b:a", "24k", "-f", "ogg", "pipe:1"],
                             input=pcm, capture_output=True, check=True).stdout
        args = ["ffmpeg", "-v", "error", "-i", "pipe:0"]
        pcm = enc
    out = subprocess.run(args + ["-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"], input=pcm,
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def evaluate(models: list[str], threads: int = 4) -> dict:
    from fdagent.providers.local_piper_tts import PiperTTS
    from fdagent.voice.local_whisper import WhisperTranscriber

    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass
    piper = PiperTTS()
    voice = piper.load()
    clips = {}
    for text, _ in PHRASES:
        pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(text))
        rate = voice.config.sample_rate
        clips[text] = {"clean": _to_16k(pcm, rate, False), "opus": _to_16k(pcm, rate, True)}
    report = {}
    for model in models:
        t = WhisperTranscriber(model)
        t.load()
        t.transcribe(clips[PHRASES[0][0]]["clean"])  # warm-up
        for cond in ("clean", "opus"):
            hits, wers, lat, misses = 0, [], [], []
            for text, keys in PHRASES:
                s = time.monotonic()
                hyp = t.transcribe(clips[text][cond])
                lat.append(time.monotonic() - s)
                ok = all(key_hit(hyp, k) for k in keys)
                hits += ok
                wers.append(wer(text, hyp))
                if not ok:
                    misses.append((text, hyp))
            report[(model, cond)] = {
                "key_accuracy": round(hits / len(PHRASES), 2), "mean_wer": round(statistics.mean(wers), 3),
                "latency_median_s": round(statistics.median(lat), 2), "latency_max_s": round(max(lat), 2),
                "misses": misses,
            }
    return report


@pytest.mark.skipif(os.getenv("FDAGENT_RUN_STT_REGRESSION") != "1", reason="opt-in: needs local models")
def test_selected_whisper_model_hears_key_values():
    model = os.getenv("FDAGENT_WHISPER_MODEL", "openai/whisper-tiny.en")
    rep = evaluate([model])
    assert rep[(model, "opus")]["key_accuracy"] >= 0.8, rep


def test_scoring_helpers():
    assert key_hit("Track order ZX 4471 please", ("zx4471",))
    assert key_hit("card number k five five two one nine", ("k55219",))
    assert not key_hit("garden hoes", ("garden hose",))
    assert key_hit("Please convert €750 to yen", ("euro", "€"))
    assert wer("a b c", "a b c") == 0.0 and wer("a b c", "a x c") == pytest.approx(1 / 3)


if __name__ == "__main__":
    models = sys.argv[1:] or ["openai/whisper-tiny.en", "openai/whisper-base.en"]
    rep = evaluate(models, threads=int(os.getenv("FDAGENT_TORCH_THREADS", "4")))
    for (model, cond), r in rep.items():
        print(f"{model:<26} {cond:<5} key_acc={r['key_accuracy']:.2f} wer={r['mean_wer']:.3f} "
              f"lat_median={r['latency_median_s']}s lat_max={r['latency_max_s']}s")
        for text, hyp in r["misses"]:
            print(f"    MISS  said={text!r}\n          heard={hyp!r}")
