# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.deps import COOKIE_NAME, decode_token, build_whisper_draft_recognizer
from backend.services.audio_session import get_audio_session
from clinvoice_audio_ingest import decode_audio_chunk, resample_pcm_s16le_mono
from clinvoice_audio_utils import TARGET_SAMPLE_RATE_HZ

router = APIRouter(tags=["websocket"])


def _valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(str(s))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _ws_user_id(websocket: WebSocket) -> int | None:
    token = websocket.cookies.get(COOKIE_NAME) or (websocket.query_params.get("token") or "").strip()
    if not token:
        return None
    try:
        data = decode_token(token)
        return int(data.get("sub", 0))
    except Exception:
        return None


@router.websocket("/audio")
async def audio_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    user_id = _ws_user_id(websocket)
    if not user_id:
        await websocket.close(code=4401)
        return

    cid = (websocket.query_params.get("consultation_id") or "").strip()
    if not _valid_uuid(cid):
        await websocket.send_json({"type": "error", "message": "consultation_id обязателен (uuid)"})
        await websocket.close(code=4400)
        return

    import clinvoice_db

    db_path = str(websocket.app.state.db_path)
    if not clinvoice_db.consultation_exists(db_path, cid, user_id):
        await websocket.send_json({"type": "error", "message": "Консультация не найдена"})
        await websocket.close(code=4404)
        return

    sess = get_audio_session(cid)
    sess.ensure_live_draft(build_whisper_draft_recognizer(websocket.app), overlap_bytes=0)

    last_draft = ""
    last_err = None

    async def push_loop() -> None:
        nonlocal last_draft, last_err
        while True:
            await asyncio.sleep(0.35)
            try:
                d = sess.draft_text()
                e = sess.draft_error()
                if d != last_draft:
                    last_draft = d
                    await websocket.send_json({"type": "draft", "text": d})
                if e != last_err:
                    last_err = e
                    if e:
                        await websocket.send_json({"type": "draft_error", "message": e})
            except Exception:
                break

    pusher = asyncio.create_task(push_loop())
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] == "websocket.receive":
                data = msg.get("text")
                if data is not None:
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict) and payload.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                else:
                    b = msg.get("bytes")
                    if b:
                        try:
                            raw, sr = decode_audio_chunk(b)
                            pcm = resample_pcm_s16le_mono(raw, sr, TARGET_SAMPLE_RATE_HZ)
                            sess.append_pcm(pcm)
                        except ValueError as ex:
                            await websocket.send_json({"type": "error", "message": str(ex)})
    except WebSocketDisconnect:
        pass
    finally:
        pusher.cancel()
        try:
            await pusher
        except asyncio.CancelledError:
            pass
