# -*- coding: utf-8 -*-
"""WebRTC → PCM; live-распознавание через Yandex SpeechKit (чанки REST v1)."""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import av
from streamlit_webrtc import AudioProcessorBase

from webrtc_draft import (
    _max_pcm_bytes,
    audio_frames_to_pcm_mono_s16le_16k,
    make_audio_resampler_s16_mono_16k,
)
from yandex_speechkit_stt import recognize_lpcm16k_mono_chunk


class SpeechKitLiveAudioProcessor(AudioProcessorBase):
    """Накапливает PCM и периодически шлёт чанки в SpeechKit; результат в live_speechkit_text."""

    def __init__(
        self,
        shared: Dict[str, Any],
        folder_id: str,
        recognize: Callable[[bytes, str], str],
        interval_sec: float,
        max_segment_bytes: int,
        min_new_bytes: int,
        overlap_bytes: int = 0,
    ) -> None:
        super().__init__()
        self._shared = shared
        self._folder_id = folder_id
        self._recognize = recognize
        self._interval = max(0.8, float(interval_sec))
        self._max_segment_bytes = max(3200, int(max_segment_bytes))
        self._min_new_bytes = max(1, int(min_new_bytes))
        self._overlap_bytes = max(0, int(overlap_bytes))
        self._stop = threading.Event()
        self._resampler: Optional[object] = None
        lk = shared.get("lock")
        if lk:
            with lk:
                shared["speechkit_pcm_committed"] = 0
                shared["live_speechkit_text"] = ""
                shared["live_speechkit_error"] = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _append_pcm(self, pcm: bytes) -> None:
        lk = self._shared.get("lock")
        if not lk or not pcm:
            return
        with lk:
            if self._shared.get("recording_paused"):
                return
        max_b = _max_pcm_bytes()
        with lk:
            acc = self._shared.setdefault("pcm_accum", bytearray())
            if len(acc) + len(pcm) > max_b:
                overflow = len(acc) + len(pcm) - max_b
                del acc[:overflow]
                c0 = int(self._shared.get("speechkit_pcm_committed") or 0)
                self._shared["speechkit_pcm_committed"] = max(0, c0 - overflow)
            acc.extend(pcm)

    def _loop(self) -> None:
        lk = self._shared.get("lock")
        if not lk:
            return
        while not self._stop.is_set():
            if self._stop.wait(self._interval):
                break
            with lk:
                paused = bool(self._shared.get("recording_paused"))
                pcm = bytes(self._shared.get("pcm_accum") or b"")
                committed = int(self._shared.get("speechkit_pcm_committed") or 0)
            if paused:
                continue
            n = len(pcm)
            if n - committed < self._min_new_bytes:
                continue
            take = min(n - committed, self._max_segment_bytes)
            ov = min(self._overlap_bytes, committed)
            start = committed - ov
            chunk = pcm[start : committed + take]
            try:
                text = self._recognize(chunk, self._folder_id).strip()
            except Exception as e:
                with lk:
                    self._shared["live_speechkit_error"] = str(e)
                continue
            if self._stop.is_set():
                break
            with lk:
                self._shared["live_speechkit_error"] = None
                prev = (self._shared.get("live_speechkit_text") or "").strip()
                t = (text or "").strip()
                if t:
                    if prev:
                        self._shared["live_speechkit_text"] = (prev + " " + t).strip()
                    else:
                        self._shared["live_speechkit_text"] = t
                self._shared["speechkit_pcm_committed"] = committed + take

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
