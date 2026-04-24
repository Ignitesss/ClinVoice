# -*- coding: utf-8 -*-
"""Обработчик аудио WebRTC → PCM → SpeechKit (черновик)."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import av
from streamlit_webrtc import AudioProcessorBase

from speechkit_stream import (
    SpeechKitStreamRecognizer,
    TARGET_SAMPLE_RATE_HZ,
    audio_frames_to_pcm_mono_s16le_16k,
    make_audio_resampler_s16_mono_16k,
    speechkit_stt_configured,
)


class DraftAudioProcessor(AudioProcessorBase):
    """Принимает очередь AudioFrame, шлёт LINEAR16 16 kHz mono в SpeechKit."""

    def __init__(
        self,
        shared: Dict[str, Any],
        api_key: Optional[str],
        folder_id: Optional[str],
        iam_token: Optional[str],
    ) -> None:
        super().__init__()
        self._shared = shared
        self._api_key = api_key
        self._folder_id = folder_id
        self._iam_token = iam_token
        self._lock = threading.Lock()
        self._recognizer: Optional[SpeechKitStreamRecognizer] = None
        self._resampler = None

    def _set_draft(self, text: str) -> None:
        lk = self._shared.get("lock")
        if lk:
            with lk:
                self._shared["draft"] = text
                self._shared["error"] = None
        else:
            self._shared["draft"] = text

    def _set_error(self, msg: str) -> None:
        lk = self._shared.get("lock")
        if lk:
            with lk:
                self._shared["error"] = msg
        else:
            self._shared["error"] = msg

    async def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        if not frames:
            return frames
        if not speechkit_stt_configured(self._api_key, self._iam_token):
            return frames
        if self._resampler is None:
            self._resampler = make_audio_resampler_s16_mono_16k()
        pcm = audio_frames_to_pcm_mono_s16le_16k(frames, self._resampler)
        if not pcm:
            return frames

        with self._lock:
            if self._recognizer is None:
                self._recognizer = SpeechKitStreamRecognizer(
                    sample_rate_hertz=TARGET_SAMPLE_RATE_HZ,
                    api_key=self._api_key,
                    iam_token=self._iam_token,
                    folder_id=self._folder_id,
                    on_display_text=self._set_draft,
                    on_error=self._set_error,
                )
                self._recognizer.start()
            rec = self._recognizer

        rec.push_pcm(pcm)
        return frames

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        return frame

    def on_ended(self) -> None:
        with self._lock:
            if self._recognizer is not None:
                self._recognizer.stop()
                self._recognizer = None
            self._resampler = None
