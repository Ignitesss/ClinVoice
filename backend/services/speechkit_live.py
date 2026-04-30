# -*- coding: utf-8 -*-
"""Фоновый цикл Yandex SpeechKit по накопленному PCM (как speechkit_webrtc, без WebRTC)."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

from clinvoice_audio_utils import max_pcm_bytes

log = logging.getLogger(__name__)


def resolve_live_speechkit_interval_sec() -> float:
    raw = (os.environ.get("CLINVOICE_LIVE_WHISPER_INTERVAL_SEC") or "").strip()
    default = 5.0
    if not raw:
        v = default
    else:
        try:
            v = float(raw)
        except ValueError:
            v = default
    return max(3.0, min(15.0, v))


def resolve_draft_tail_max_seconds() -> float:
    raw = (os.environ.get("CLINVOICE_DRAFT_TAIL_MAX_SECONDS") or "").strip()
    default = 22.0
    if not raw:
        v = default
    else:
        try:
            v = float(raw)
        except ValueError:
            v = default
    return max(5.0, min(60.0, v))


def resolve_draft_min_new_seconds() -> float:
    raw = (os.environ.get("CLINVOICE_DRAFT_MIN_NEW_SECONDS") or "").strip()
    default = 1.0
    if not raw:
        v = default
    else:
        try:
            v = float(raw)
        except ValueError:
            v = default
    return max(0.2, min(5.0, v))


def resolve_draft_tail_overlap_sec() -> float:
    raw = (os.environ.get("CLINVOICE_DRAFT_TAIL_OVERLAP_SEC") or "").strip()
    if not raw:
        return 0.35
    try:
        v = float(raw)
    except ValueError:
        return 0.35
    return max(0.0, min(3.0, v))


class SpeechKitBackgroundLoop:
    """Пишет в shared: live_speechkit_text, live_speechkit_error, speechkit_pcm_committed."""

    def __init__(
        self,
        shared: Dict[str, Any],
        folder_id: str,
        recognize: Callable[[bytes, str], str],
        interval_sec: Optional[float] = None,
    ) -> None:
        self._shared = shared
        self._folder_id = folder_id
        self._recognize = recognize
        iv = interval_sec if interval_sec is not None else resolve_live_speechkit_interval_sec()
        self._interval = max(0.8, float(iv))
        self._max_segment_bytes = int(resolve_draft_tail_max_seconds() * 32000)
        self._min_new_bytes = int(resolve_draft_min_new_seconds() * 32000)
        self._overlap_bytes = int(resolve_draft_tail_overlap_sec() * 32000)
        self._stop = threading.Event()
        lk = shared.get("lock")
        if lk:
            with lk:
                shared["speechkit_pcm_committed"] = 0
                shared["live_speechkit_text"] = ""
                shared["live_speechkit_error"] = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

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
                log.warning("SpeechKit: %s", e)
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
