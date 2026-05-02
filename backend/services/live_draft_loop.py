# -*- coding: utf-8 -*-
"""Фоновый цикл «живого» черновика по накопленному PCM (Whisper по чанкам)."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


def resolve_live_draft_interval_sec() -> float:
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


def resolve_draft_tail_overlap_bytes() -> int:
    """Для облачного STT с перекрытием окон; для Whisper по умолчанию 0 (без дублей в тексте)."""
    raw = (os.environ.get("CLINVOICE_DRAFT_TAIL_OVERLAP_SEC") or "").strip()
    if not raw:
        return 0
    try:
        v = float(raw)
    except ValueError:
        return 0
    sec = max(0.0, min(3.0, v))
    return int(sec * 32000)


class LiveDraftBackgroundLoop:
    """Пишет в shared: live_draft_text, live_draft_error, draft_pcm_committed."""

    def __init__(
        self,
        shared: Dict[str, Any],
        recognize: Callable[[bytes, str], str],
        *,
        interval_sec: Optional[float] = None,
        overlap_bytes: Optional[int] = None,
    ) -> None:
        self._shared = shared
        self._recognize = recognize
        iv = interval_sec if interval_sec is not None else resolve_live_draft_interval_sec()
        self._interval = max(0.8, float(iv))
        self._max_segment_bytes = int(resolve_draft_tail_max_seconds() * 32000)
        self._min_new_bytes = int(resolve_draft_min_new_seconds() * 32000)
        ov = resolve_draft_tail_overlap_bytes() if overlap_bytes is None else int(overlap_bytes)
        self._overlap_bytes = max(0, ov)
        self._stop = threading.Event()
        lk = shared.get("lock")
        if lk:
            with lk:
                shared["draft_pcm_committed"] = 0
                shared["live_draft_text"] = ""
                shared["live_draft_error"] = None
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
                committed = int(self._shared.get("draft_pcm_committed") or 0)
            if paused:
                continue
            n = len(pcm)
            if n - committed < self._min_new_bytes:
                continue
            take = min(n - committed, self._max_segment_bytes)
            ov = min(self._overlap_bytes, committed)
            start = max(0, committed - ov)
            chunk = pcm[start : committed + take]
            prev = ""
            with lk:
                prev = (self._shared.get("live_draft_text") or "").strip()
            try:
                text = self._recognize(chunk, prev).strip()
            except Exception as e:
                log.warning("Live draft ASR: %s", e)
                with lk:
                    self._shared["live_draft_error"] = str(e)
                continue
            if self._stop.is_set():
                break
            with lk:
                self._shared["live_draft_error"] = None
                t = (text or "").strip()
                if t:
                    if prev:
                        self._shared["live_draft_text"] = (prev + " " + t).strip()
                    else:
                        self._shared["live_draft_text"] = t
                self._shared["draft_pcm_committed"] = committed + take
