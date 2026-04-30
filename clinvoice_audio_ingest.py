# -*- coding: utf-8 -*-
"""Декодирование бинарных кадров WS: WAV mono s16le или сырой PCM."""

from __future__ import annotations

import io
import wave
from typing import Tuple

import numpy as np

from clinvoice_audio_utils import TARGET_SAMPLE_RATE_HZ


def decode_audio_chunk(data: bytes) -> Tuple[bytes, int]:
    """
    Возвращает (pcm_s16le_mono, sample_rate_hz).
    Сырой фрейм без RIFF считается уже 16 kHz mono s16le.
    """
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        with wave.open(io.BytesIO(data), "rb") as w:
            ch = w.getnchannels()
            sw = w.getsampwidth()
            fr = w.getframerate()
            nframes = w.getnframes()
            raw = w.readframes(nframes)
        if ch != 1 or sw != 2:
            raise ValueError("Ожидается моно 16-bit WAV")
        return raw, int(fr)
    return data, TARGET_SAMPLE_RATE_HZ


def resample_pcm_s16le_mono(pcm: bytes, src_sr: int, dst_sr: int = TARGET_SAMPLE_RATE_HZ) -> bytes:
    if src_sr == dst_sr or not pcm:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    n = len(x)
    if n == 0:
        return b""
    n_out = max(1, int(round(n * float(dst_sr) / float(src_sr))))
    t_src = np.arange(n, dtype=np.float64)
    t_dst = np.linspace(0.0, float(n - 1), num=n_out, endpoint=True)
    y = np.interp(t_dst, t_src, x)
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return y.tobytes()
