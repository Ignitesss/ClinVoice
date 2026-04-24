# -*- coding: utf-8 -*-
"""Обработчик аудио WebRTC → PCM и инкрементальный live-Whisper по хвосту буфера."""

from __future__ import annotations

import io
import os
import threading
import wave
from typing import Any, Callable, Dict, List, Optional, Tuple

import av
from streamlit_webrtc import AudioProcessorBase

TARGET_SAMPLE_RATE_HZ = 16000

_DEFAULT_MAX_PCM = 80 * 1024 * 1024


def _max_pcm_bytes() -> int:
    raw = (os.environ.get("CLINVOICE_MAX_PCM_BYTES") or "").strip()
    if raw.isdigit():
        return max(1_000_000, int(raw))
    return _DEFAULT_MAX_PCM


def audio_frames_to_pcm_mono_s16le_16k(
    frames: List[object],
    resampler: object,
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


class DraftAudioProcessor(AudioProcessorBase):
    """Принимает очередь AudioFrame: накапливает PCM и инкрементально дополняет live_whisper_text."""

    def __init__(
        self,
        shared: Dict[str, Any],
        transcribe_pcm: Callable[[bytes], Tuple[str, Optional[str]]],
        interval_sec: float,
        max_segment_bytes: int,
        min_new_bytes: int,
        overlap_bytes: int = 0,
    ) -> None:
        super().__init__()
        self._shared = shared
        self._transcribe_pcm = transcribe_pcm
        self._interval = max(1.0, float(interval_sec))
        self._max_segment_bytes = max(3200, int(max_segment_bytes))
        self._min_new_bytes = max(1, int(min_new_bytes))
        self._overlap_bytes = max(0, int(overlap_bytes))
        self._stop = threading.Event()
        self._resampler: Optional[object] = None
        lk = shared.get("lock")
        if lk:
            with lk:
                shared["live_draft_pcm_committed"] = 0
                shared["live_whisper_text"] = ""
                shared["live_whisper_error"] = None
                shared.pop("live_whisper_last_processed_pcm_len", None)
        self._thread = threading.Thread(target=self._live_whisper_loop, daemon=True)
        self._thread.start()

    def _append_pcm(self, pcm: bytes) -> None:
        lk = self._shared.get("lock")
        if not lk or not pcm:
            return
        max_b = _max_pcm_bytes()
        with lk:
            acc = self._shared.setdefault("pcm_accum", bytearray())
            if len(acc) + len(pcm) > max_b:
                overflow = len(acc) + len(pcm) - max_b
                del acc[:overflow]
            acc.extend(pcm)

    def _live_whisper_loop(self) -> None:
        lk = self._shared.get("lock")
        if not lk:
            return
        while not self._stop.is_set():
            if self._stop.wait(self._interval):
                break
            with lk:
                pcm = bytes(self._shared.get("pcm_accum") or b"")
                committed = int(self._shared.get("live_draft_pcm_committed") or 0)
            n = len(pcm)
            if n - committed < self._min_new_bytes:
                continue
            take = min(n - committed, self._max_segment_bytes)
            ov = min(self._overlap_bytes, committed)
            start = committed - ov
            chunk = pcm[start : committed + take]
            text, err = self._transcribe_pcm(chunk)
            if self._stop.is_set():
                break
            with lk:
                if err:
                    self._shared["live_whisper_error"] = err
                    continue
                self._shared["live_whisper_error"] = None
                prev = (self._shared.get("live_whisper_text") or "").strip()
                t = (text or "").strip()
                if t:
                    if prev:
                        self._shared["live_whisper_text"] = (prev + " " + t).strip()
                    else:
                        self._shared["live_whisper_text"] = t
                self._shared["live_draft_pcm_committed"] = committed + take

    async def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        if not frames:
            return frames
        if self._resampler is None:
            self._resampler = make_audio_resampler_s16_mono_16k()
        pcm = audio_frames_to_pcm_mono_s16le_16k(frames, self._resampler)
        if not pcm:
            return frames
        self._append_pcm(pcm)
        return frames

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        return frame

    def on_ended(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=60.0)
        self._resampler = None
