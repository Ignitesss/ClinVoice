# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import clinvoice_db
from backend.deps import get_current_user_id, get_db_path
from backend.settings import resolve_protocol_delay_sec
from clinvoice_protocol_io import (
    build_structured_protocol_txt,
    format_consultation_date_gmt3,
)
from clinvoice_transcript_clean import strip_whisper_tv_caption_artifacts
from protocol import fill_protocol_from_transcript, format_protocol_editor_text
from backend.services.audio_session import drop_audio_session, get_audio_session

router = APIRouter(prefix="/consultations", tags=["consultations"])


def _valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(str(s))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _empty_snapshot() -> Dict[str, Any]:
    return {
        "live_transcript_editor": "",
        "live_transcript_pause_auto_sync": False,
        "original_transcription": None,
        "doctor_transcript_editor": "",
        "protocol_editor_text": "",
        "protocol_consultation_date": "",
        "status": "draft",
    }


def _merge_snapshot(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if k in _empty_snapshot() or k == "status":
            out[k] = v
    return out


class SnapshotPatch(BaseModel):
    live_transcript_editor: Optional[str] = None
    live_transcript_pause_auto_sync: Optional[bool] = None
    original_transcription: Optional[str] = None
    doctor_transcript_editor: Optional[str] = None
    protocol_editor_text: Optional[str] = None
    protocol_consultation_date: Optional[str] = None
    status: Optional[str] = None

    def as_patch(self) -> Dict[str, Any]:
        d = self.model_dump(exclude_unset=True)
        return d


@router.post("")
def create_consultation(
    user_id: int = Depends(get_current_user_id),
    db_path: str = Depends(get_db_path),
) -> dict:
    cid = str(uuid.uuid4())
    clinvoice_db.upsert_consultation_row(db_path, cid, user_id, "draft")
    return {"id": cid}


@router.get("/{consultation_id}")
def get_consultation(
    consultation_id: str,
    user_id: int = Depends(get_current_user_id),
    db_path: str = Depends(get_db_path),
) -> dict:
    if not _valid_uuid(consultation_id):
        raise HTTPException(status_code=400, detail="Некорректный идентификатор")
    row = clinvoice_db.get_consultation(db_path, consultation_id, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Консультация не найдена")
    snap = clinvoice_db.load_latest_snapshot(db_path, consultation_id, user_id)
    return {
        "id": row["id"],
        "created_at": int(row["created_at"]),
        "updated_at": int(row["updated_at"]),
        "status": str(row["status"]),
        "snapshot": snap,
    }


@router.get("/{consultation_id}/snapshot")
def get_snapshot(
    consultation_id: str,
    user_id: int = Depends(get_current_user_id),
    db_path: str = Depends(get_db_path),
) -> dict:
    if not _valid_uuid(consultation_id):
        raise HTTPException(status_code=400, detail="Некорректный идентификатор")
    if not clinvoice_db.consultation_exists(db_path, consultation_id, user_id):
        raise HTTPException(status_code=404, detail="Консультация не найдена")
    snap = clinvoice_db.load_latest_snapshot(db_path, consultation_id, user_id)
    if not snap:
        return {"snapshot": _empty_snapshot()}
    merged = _merge_snapshot(_empty_snapshot(), snap)
    return {"snapshot": merged}


@router.put("/{consultation_id}/snapshot")
def put_snapshot(
    consultation_id: str,
    body: SnapshotPatch,
    user_id: int = Depends(get_current_user_id),
    db_path: str = Depends(get_db_path),
) -> dict:
    if not _valid_uuid(consultation_id):
        raise HTTPException(status_code=400, detail="Некорректный идентификатор")
    if not clinvoice_db.consultation_exists(db_path, consultation_id, user_id):
        raise HTTPException(status_code=404, detail="Консультация не найдена")
    prev = clinvoice_db.load_latest_snapshot(db_path, consultation_id, user_id) or {}
    base = _merge_snapshot(_empty_snapshot(), prev)
    patch = body.as_patch()
    merged = _merge_snapshot(base, patch)
    merged.setdefault("status", "draft")
    clinvoice_db.save_draft_snapshot(db_path, consultation_id, user_id, merged)
    return {"ok": True, "snapshot": merged}


@router.post("/{consultation_id}/reset-audio")
def reset_audio(
    consultation_id: str,
    user_id: int = Depends(get_current_user_id),
    db_path: str = Depends(get_db_path),
) -> dict:
    if not _valid_uuid(consultation_id):
        raise HTTPException(status_code=400, detail="Некорректный идентификатор")
    if not clinvoice_db.consultation_exists(db_path, consultation_id, user_id):
        raise HTTPException(status_code=404, detail="Консультация не найдена")
    sess = get_audio_session(consultation_id)
    sess.clear_buffer()
    return {"ok": True}


@router.post("/{consultation_id}/finalize")
async def finalize(
    consultation_id: str,
    user_id: int = Depends(get_current_user_id),
    db_path: str = Depends(get_db_path),
) -> dict:
    if not _valid_uuid(consultation_id):
        raise HTTPException(status_code=400, detail="Некорректный идентификатор")
    if not clinvoice_db.consultation_exists(db_path, consultation_id, user_id):
        raise HTTPException(status_code=404, detail="Консультация не найдена")

    sess = get_audio_session(consultation_id)
    pcm = sess.copy_pcm()
    if len(pcm) < 3200:
        raise HTTPException(status_code=400, detail="Слишком короткая запись для Whisper")

    whisper_txt = strip_whisper_tv_caption_artifacts(sess.draft_text())
    if not whisper_txt:
        raise HTTPException(
            status_code=400,
            detail="Черновик транскрипта пуст. Дождитесь появления текста во время записи или проверьте микрофон.",
        )

    await asyncio.sleep(resolve_protocol_delay_sec())

    def _protocol():
        return fill_protocol_from_transcript(whisper_txt)

    try:
        protocol = await asyncio.to_thread(_protocol)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка заполнения протокола: {e}") from e

    consultation_date = format_consultation_date_gmt3()
    editor_text = format_protocol_editor_text(consultation_date, protocol)

    prev = clinvoice_db.load_latest_snapshot(db_path, consultation_id, user_id) or {}
    merged = _merge_snapshot(_empty_snapshot(), prev)
    merged["live_transcript_editor"] = whisper_txt
    merged["original_transcription"] = whisper_txt
    merged["doctor_transcript_editor"] = whisper_txt
    merged["protocol_consultation_date"] = consultation_date
    merged["protocol_editor_text"] = editor_text
    merged["status"] = "draft"
    clinvoice_db.save_draft_snapshot(db_path, consultation_id, user_id, merged)

    sess.clear_buffer()

    protocol_txt = build_structured_protocol_txt(protocol, consultation_date)
    return {
        "transcription": whisper_txt,
        "protocol": protocol,
        "protocol_editor_text": editor_text,
        "protocol_consultation_date": consultation_date,
        "transcript_txt": whisper_txt,
        "protocol_txt": protocol_txt,
    }


@router.delete("/{consultation_id}/session")
def delete_session(consultation_id: str, user_id: int = Depends(get_current_user_id), db_path: str = Depends(get_db_path)) -> dict:
    """Очистить in-memory аудио-сессию (опционально)."""
    if not _valid_uuid(consultation_id):
        raise HTTPException(status_code=400, detail="Некорректный идентификатор")
    if not clinvoice_db.consultation_exists(db_path, consultation_id, user_id):
        raise HTTPException(status_code=404, detail="Консультация не найдена")
    drop_audio_session(consultation_id)
    return {"ok": True}
