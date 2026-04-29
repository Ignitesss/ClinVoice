# -*- coding: utf-8 -*-
"""Yandex SpeechKit STT v1: распознавание сырого LPCM 16-bit mono 16 kHz (чанк)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from protocol import resolve_yandex_api_key, resolve_yandex_folder_id, resolve_yandex_iam_token

_STT_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"


def _stt_headers(folder_id: str) -> dict:
    iam = resolve_yandex_iam_token()
    if iam:
        return {
            "Authorization": f"Bearer {iam}",
            "x-folder-id": folder_id,
            "Content-Type": "application/octet-stream",
        }
    key = resolve_yandex_api_key()
    if not key:
        raise RuntimeError("Не задан YANDEX_CLOUD_API_KEY или YANDEX_IAM_TOKEN для SpeechKit")
    return {
        "Authorization": f"Api-Key {key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/octet-stream",
    }


def speechkit_configured() -> bool:
    fid = resolve_yandex_folder_id()
    if not fid:
        return False
    return bool(resolve_yandex_api_key() or resolve_yandex_iam_token())


def recognize_lpcm16k_mono_chunk(pcm_s16le: bytes, folder_id: Optional[str] = None) -> str:
    """
    Один запрос к SpeechKit v1: тело — сырые сэмплы s16le mono, 16000 Hz.
    Возвращает распознанную строку (может быть пустой).
    """
    if not pcm_s16le:
        return ""
    fid = (folder_id or resolve_yandex_folder_id() or "").strip()
    if not fid:
        raise RuntimeError("Не задан YANDEX_FOLDER_ID")
    topic = (os.environ.get("CLINVOICE_SPEECHKIT_TOPIC") or "general").strip() or "general"
    q = urllib.parse.urlencode(
        {
            "folderId": fid,
            "lang": "ru-RU",
            "format": "lpcm",
            "sampleRateHertz": "16000",
            "topic": topic,
        }
    )
    url = f"{_STT_URL}?{q}"
    data = pcm_s16le
    req = urllib.request.Request(url, data=data, headers=_stt_headers(fid), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SpeechKit HTTP {e.code}: {err}") from e
    if isinstance(body, dict):
        if body.get("error_code"):
            raise RuntimeError(str(body.get("error_message") or body))
        return str(body.get("result") or "").strip()
    return ""
