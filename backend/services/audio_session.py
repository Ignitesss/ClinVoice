# -*- coding: utf-8 -*-
"""Состояние консультации в памяти: PCM, фоновый цикл живого черновика (Whisper)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from clinvoice_audio_utils import max_pcm_bytes

if TYPE_CHECKING:
    from backend.services.live_draft_loop import LiveDraftBackgroundLoop
    from clinvoice_asr import AudioTranscriberWithMetrics


def _empty_shared() -> Dict[str, Any]:
    return {
        "lock": threading.Lock(),
        "pcm_accum": bytearray(),
        "draft_pcm_committed": 0,
        "live_draft_text": "",
        "live_draft_error": None,
    }


class ConsultationAudioSession:
    def __init__(self, consultation_id: str) -> None:
        self.consultation_id = consultation_id
        self.shared = _empty_shared()
        self._draft_loop: Optional["LiveDraftBackgroundLoop"] = None

    def ensure_live_draft(
        self,
        recognize: Callable[[bytes, str], str],
        *,
        overlap_bytes: int = 0,
        clear_state: bool = True,
    ) -> None:
        if self._draft_loop is not None:
            return
        from backend.services.live_draft_loop import LiveDraftBackgroundLoop

        self._draft_loop = LiveDraftBackgroundLoop(
            self.shared,
            recognize,
            overlap_bytes=overlap_bytes,
            clear_state=clear_state,
        )

    def stop_live_draft(self) -> None:
        if self._draft_loop is not None:
            self._draft_loop.stop()
            self._draft_loop = None

    def append_pcm(self, pcm: bytes) -> None:
        if not pcm:
            return
        lk = self.shared["lock"]
        with lk:
            acc: bytearray = self.shared.setdefault("pcm_accum", bytearray())
            cap = max_pcm_bytes()
            if len(acc) + len(pcm) > cap:
                overflow = len(acc) + len(pcm) - cap
                del acc[:overflow]
                c0 = int(self.shared.get("draft_pcm_committed") or 0)
                self.shared["draft_pcm_committed"] = max(0, c0 - overflow)
            acc.extend(pcm)

    def flush_pending_whisper_draft(self, transcriber: "AudioTranscriberWithMetrics") -> None:
        """Догнать draft_pcm_committed до конца PCM (после остановки фонового цикла)."""
        from clinvoice_asr import WHISPER_DYNAMIC_INITIAL_PROMPT_MAX_CHARS, transcribe_pcm_s16le_mono
        from backend.services.live_draft_loop import resolve_draft_tail_max_seconds

        max_bytes = int(resolve_draft_tail_max_seconds() * 32000)
        lk = self.shared["lock"]
        while True:
            with lk:
                pcm = bytes(self.shared.get("pcm_accum") or b"")
                committed = int(self.shared.get("draft_pcm_committed") or 0)
            n = len(pcm)
            if committed >= n:
                return
            take = min(n - committed, max_bytes)
            chunk = pcm[committed : committed + take]
            with lk:
                prev = (self.shared.get("live_draft_text") or "").strip()
            p = prev
            ip = p[-WHISPER_DYNAMIC_INITIAL_PROMPT_MAX_CHARS:] if p else None
            text = transcribe_pcm_s16le_mono(
                transcriber,
                chunk,
                language="ru",
                draft=True,
                initial_prompt=ip,
            ).strip()
            with lk:
                cur_prev = (self.shared.get("live_draft_text") or "").strip()
                t = (text or "").strip()
                if t:
                    if cur_prev:
                        self.shared["live_draft_text"] = (cur_prev + " " + t).strip()
                    else:
                        self.shared["live_draft_text"] = t
                self.shared["draft_pcm_committed"] = committed + take

    def clear_buffer(self) -> None:
        with self.shared["lock"]:
            self.shared["pcm_accum"] = bytearray()
            self.shared["live_draft_text"] = ""
            self.shared["live_draft_error"] = None
            self.shared["draft_pcm_committed"] = 0

    def copy_pcm(self) -> bytes:
        with self.shared["lock"]:
            return bytes(self.shared.get("pcm_accum") or b"")

    def draft_text(self) -> str:
        with self.shared["lock"]:
            return (self.shared.get("live_draft_text") or "").strip()

    def draft_error(self) -> Optional[str]:
        with self.shared["lock"]:
            e = self.shared.get("live_draft_error")
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
        s.stop_live_draft()
