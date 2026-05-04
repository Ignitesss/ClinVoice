# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import threading
import time
from typing import Annotated, Callable, Optional

import jwt
from fastapi import Cookie, Depends, HTTPException, Request, status

import auth
import clinvoice_db
from backend.settings import get_settings
from clinvoice_asr import (
    WHISPER_DYNAMIC_INITIAL_PROMPT_MAX_CHARS,
    AudioTranscriberWithMetrics,
    resolve_hub_model_id,
    transcribe_pcm_s16le_mono,
)

log = logging.getLogger(__name__)

COOKIE_NAME = "clinvoice_token"
ALGORITHM = "HS256"


def _jwt_secret() -> str:
    s = (get_settings().clinvoice_jwt_secret or "").strip()
    if s:
        return s
    log.warning("CLINVOICE_JWT_SECRET не задан — используется небезопасный ключ разработки")
    return "clinvoice-dev-only-change-me"


def create_access_token(*, user_id: int, username: str) -> str:
    exp = int(time.time()) + int(get_settings().clinvoice_jwt_expire_hours) * 3600
    payload = {"sub": str(user_id), "username": username, "exp": exp}
    return jwt.encode(payload, _jwt_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, _jwt_secret(), algorithms=[ALGORITHM])


def get_db_path(request: Request) -> str:
    p = getattr(request.app.state, "db_path", None)
    if not p:
        raise HTTPException(status_code=500, detail="БД не инициализирована")
    return str(p)


def get_current_user_id(
    request: Request,
    access_token: Annotated[Optional[str], Cookie(alias=COOKIE_NAME)] = None,
) -> int:
    token = access_token
    if not token:
        auth_h = request.headers.get("authorization") or ""
        if auth_h.lower().startswith("bearer "):
            token = auth_h[7:].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Нет сессии")
    try:
        data = decode_token(token)
        uid = int(data.get("sub", 0))
        if uid <= 0:
            raise ValueError("bad sub")
        return uid
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Сессия истекла")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверная сессия")


def get_transcriber_for_app(app) -> AudioTranscriberWithMetrics:
    lock: threading.Lock = app.state.transcriber_lock
    with lock:
        if app.state.transcriber is None:
            hub = resolve_hub_model_id()
            log.info("Загрузка ASR (hub=%s)", hub or "(openai)")
            app.state.transcriber = AudioTranscriberWithMetrics(
                model_size="small",
                hub_model_id=hub or None,
                silent_ui=True,
            )
        return app.state.transcriber


def get_transcriber(request: Request) -> AudioTranscriberWithMetrics:
    return get_transcriber_for_app(request.app)


def build_whisper_draft_recognizer(app) -> Callable[[bytes, str], str]:
    """Тот же колбэк, что и для WebSocket: черновик Whisper по куску PCM + предыдущий текст."""

    def _recognize_whisper(pcm_chunk: bytes, prev_draft: str) -> str:
        transcriber = get_transcriber_for_app(app)
        p = (prev_draft or "").strip()
        ip = p[-WHISPER_DYNAMIC_INITIAL_PROMPT_MAX_CHARS:] if p else None
        return transcribe_pcm_s16le_mono(
            transcriber,
            pcm_chunk,
            language="ru",
            draft=True,
            initial_prompt=ip,
        )

    return _recognize_whisper
