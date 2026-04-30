# -*- coding: utf-8 -*-
"""PCM s16le mono → WAV (Whisper / буферы)."""

from __future__ import annotations

import io
import os
import wave

TARGET_SAMPLE_RATE_HZ = 16000
_DEFAULT_MAX_PCM = 80 * 1024 * 1024


def max_pcm_bytes() -> int:
    raw = (os.environ.get("CLINVOICE_MAX_PCM_BYTES") or "").strip()
    if raw.isdigit():
        return max(1_000_000, int(raw))
    return _DEFAULT_MAX_PCM


def pcm_mono_s16le_to_wav_bytes(pcm: bytes, sample_rate: int = TARGET_SAMPLE_RATE_HZ) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return out.getvalue()
