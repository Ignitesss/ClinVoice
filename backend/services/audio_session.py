# -*- coding: utf-8 -*-
"""Состояние консультации в памяти: PCM, SpeechKit, ссылка на фоновый цикл."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

from clinvoice_audio_utils import max_pcm_bytes

if TYPE_CHECKING:
    from backend.services.speechkit_live import SpeechKitBackgroundLoop


def _empty_shared() -> Dict[str, Any]:
    return {
        "lock": threading.Lock(),
        "pcm_accum": bytearray(),
        "recording_paused": False,
        "speechkit_pcm_committed": 0,
        "live_speechkit_text": "",
        "live_speechkit_error": None,
    }


class ConsultationAudioSession:
    def __init__(self, consultation_id: str) -> None:
        self.consultation_id = consultation_id
        self.shared = _empty_shared()
        self._speechkit_loop: Optional[SpeechKitBackgroundLoop] = None

    def ensure_speechkit(self, folder_id: str, recognize) -> None:
        if self._speechkit_loop is not None:
            return
        from backend.services.speechkit_live import SpeechKitBackgroundLoop

        self._speechkit_loop = SpeechKitBackgroundLoop(self.shared, folder_id, recognize)

    def stop_speechkit(self) -> None:
        if self._speechkit_loop is not None:
            self._speechkit_loop.stop()
            self._speechkit_loop = None

    def append_pcm(self, pcm: bytes) -> None:
        if not pcm:
            return
        lk = self.shared["lock"]
        with lk:
            if self.shared.get("recording_paused"):
                return
            acc: bytearray = self.shared.setdefault("pcm_accum", bytearray())
            cap = max_pcm_bytes()
            if len(acc) + len(pcm) > cap:
                overflow = len(acc) + len(pcm) - cap
                del acc[:overflow]
                c0 = int(self.shared.get("speechkit_pcm_committed") or 0)
                self.shared["speechkit_pcm_committed"] = max(0, c0 - overflow)
            acc.extend(pcm)

    def set_paused(self, paused: bool) -> None:
        with self.shared["lock"]:
            self.shared["recording_paused"] = bool(paused)

    def clear_buffer(self) -> None:
        with self.shared["lock"]:
            self.shared["pcm_accum"] = bytearray()
            self.shared["live_speechkit_text"] = ""
            self.shared["live_speechkit_error"] = None
            self.shared["speechkit_pcm_committed"] = 0

    def copy_pcm(self) -> bytes:
        with self.shared["lock"]:
            return bytes(self.shared.get("pcm_accum") or b"")

    def draft_text(self) -> str:
        with self.shared["lock"]:
            return (self.shared.get("live_speechkit_text") or "").strip()

    def draft_error(self) -> Optional[str]:
        with self.shared["lock"]:
            e = self.shared.get("live_speechkit_error")
            return str(e) if e else None


_sessions_lock = threading.Lock()
_sessions: Dict[str, ConsultationAudioSession] = {}


def get_audio_session(consultation_id: str) -> ConsultationAudioSession:
    with _sessions_lock:
        s = _sessions.get(consultation_id)
        if s is None:
            s = ConsultationAudioSession(consultation_id)
            _sessions[consultation_id] = s
        return s


def drop_audio_session(consultation_id: str) -> None:
    with _sessions_lock:
        s = _sessions.pop(consultation_id, None)
    if s:
        s.stop_speechkit()
