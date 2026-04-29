# -*- coding: utf-8 -*-
"""Утилиты WebRTC → PCM (LINEAR16 mono 16 kHz) для SpeechKit и Whisper."""

from __future__ import annotations

import io
import os
import wave
from typing import Any, List

import av

TARGET_SAMPLE_RATE_HZ = 16000

_DEFAULT_MAX_PCM = 80 * 1024 * 1024


def _max_pcm_bytes() -> int:
    raw = (os.environ.get("CLINVOICE_MAX_PCM_BYTES") or "").strip()
    if raw.isdigit():
        return max(1_000_000, int(raw))
    return _DEFAULT_MAX_PCM


def audio_frames_to_pcm_mono_s16le_16k(
    frames: List[Any],
    resampler: Any,
) -> bytes:
    """PyAV AudioFrame → LINEAR16 mono 16 kHz; ``resampler`` — один AudioResampler на сеанс."""
    out_chunks: List[bytes] = []
    for frame in frames:
        for converted in resampler.resample(frame):
            arr = converted.to_ndarray()
            out_chunks.append(arr.tobytes())
    return b"".join(out_chunks)


def make_audio_resampler_s16_mono_16k():
    """Создать PyAV AudioResampler (один экземпляр на сеанс WebRTC)."""
    return av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=TARGET_SAMPLE_RATE_HZ
    )


def pcm_mono_s16le_to_wav_bytes(pcm: bytes, sample_rate: int = TARGET_SAMPLE_RATE_HZ) -> bytes:
    """Сырые PCM s16le mono → WAV (для Whisper)."""
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return out.getvalue()
