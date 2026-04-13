# -*- coding: utf-8 -*-
"""
Протокол консультации через Yandex Cloud Foundation Models (YandexGPT).

Секреты / переменные окружения:
  YANDEX_CLOUD_API_KEY — API-ключ сервисного аккаунта (или YC_API_KEY / YANDEX_API_KEY)
  YANDEX_FOLDER_ID — идентификатор каталога в Yandex Cloud (или YC_FOLDER_ID)

Опционально:
  YANDEX_MODEL_URI — полный URI, например gpt://<folder_id>/yandexgpt/latest
  YANDEX_GPT_VARIANT — если URI не задан: суффикс после folder, по умолчанию yandexgpt/latest
                       (например yandexgpt-lite/latest)
  YANDEX_IAM_TOKEN — вместо API-ключа: IAM-токен, заголовок Authorization: Bearer …

Рекомендуемая схема CSV для будущей разметки / обучения (MVP не парсит файлы):
  audio_path, category, transcript_ref, complaints, anamnesis, conclusion,
  recommendations, consultation_date_utc (опционально)

Тип задачи: генеративное структурирование (не дословное извлечение span).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

YANDEX_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

# Поля JSON после разбора ответа модели
PROTOCOL_FIELD_KEYS = ("complaints", "anamnesis", "conclusion", "recommendations")

PROTOCOL_TRAINING_CSV_COLUMNS = (
    "audio_path",
    "category",
    "transcript_ref",
    "complaints",
    "anamnesis",
    "conclusion",
    "recommendations",
    "consultation_date",
)

SYSTEM_PROMPT = """Ты помощник врача-терапевта. По тексту транскрипта консультации заполни поля медицинского протокола на русском языке.
Правила:
- Не выдумывай факты, диагнозы и назначения, которых нет в транскрипте.
- Если для раздела нет данных в транскрипте, напиши в этом поле: «Не указано».
- Формулировки — деловой медицинский стиль, кратко и по существу.
- Жалобы: симптомы и жалобы пациента.
- Анамнез: перенесённые заболевания, анамнестические данные из текста.
- Заключение: выводы врача из транскрипта.
- Рекомендации: рекомендации, назначения, режим — только если они звучат в транскрипте."""

JSON_INSTRUCTION = """
Ответь ТОЛЬКО одним JSON-объектом без пояснений до и после, без markdown и без блоков кода.
Формат (все ключи и строки на русском содержимом в значениях):
{"complaints":"текст","anamnesis":"текст","conclusion":"текст","recommendations":"текст"}
Экранируй кавычки и переносы строк внутри значений по правилам JSON.
"""


def resolve_yandex_api_key() -> Optional[str]:
    for name in ("YANDEX_CLOUD_API_KEY", "YC_API_KEY", "YANDEX_API_KEY"):
        k = (os.environ.get(name) or "").strip()
        if k:
            return k
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for key in ("YANDEX_CLOUD_API_KEY", "YC_API_KEY", "YANDEX_API_KEY"):
                if key in st.secrets:
                    return str(st.secrets[key]).strip()
    except Exception:
        pass
    return None


def resolve_yandex_iam_token() -> Optional[str]:
    t = (os.environ.get("YANDEX_IAM_TOKEN") or os.environ.get("YC_IAM_TOKEN") or "").strip()
    if t:
        return t
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for key in ("YANDEX_IAM_TOKEN", "YC_IAM_TOKEN"):
                if key in st.secrets:
                    return str(st.secrets[key]).strip()
    except Exception:
        pass
    return None


def resolve_yandex_folder_id() -> Optional[str]:
    for name in ("YANDEX_FOLDER_ID", "YC_FOLDER_ID"):
        k = (os.environ.get(name) or "").strip()
        if k:
            return k
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for key in ("YANDEX_FOLDER_ID", "YC_FOLDER_ID"):
                if key in st.secrets:
                    return str(st.secrets[key]).strip()
    except Exception:
        pass
    return None


def resolve_yandex_model_uri() -> Optional[str]:
    explicit = (os.environ.get("YANDEX_MODEL_URI") or "").strip()
    if explicit:
        return explicit
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "YANDEX_MODEL_URI" in st.secrets:
            return str(st.secrets["YANDEX_MODEL_URI"]).strip()
    except Exception:
        pass
    folder = resolve_yandex_folder_id()
    if not folder:
        return None
    variant = (os.environ.get("YANDEX_GPT_VARIANT") or "yandexgpt/latest").strip()
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "YANDEX_GPT_VARIANT" in st.secrets:
            variant = str(st.secrets["YANDEX_GPT_VARIANT"]).strip() or variant
    except Exception:
        pass
    return f"gpt://{folder}/{variant}"


def yandex_llm_configured() -> bool:
    folder = resolve_yandex_folder_id()
    if not folder:
        return False
    if resolve_yandex_api_key() or resolve_yandex_iam_token():
        return True
    return False


def _auth_headers(folder_id: str) -> Dict[str, str]:
    iam = resolve_yandex_iam_token()
    if iam:
        return {
            "Authorization": f"Bearer {iam}",
            "x-folder-id": folder_id,
            "Content-Type": "application/json",
        }
    key = resolve_yandex_api_key()
    if not key:
        raise RuntimeError("Не задан ключ доступа или IAM-токен в настройках")
    return {
        "Authorization": f"Api-Key {key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/json",
    }


def _post_yandex_completion(body: Dict[str, Any], folder_id: str) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        YANDEX_COMPLETION_URL,
        data=data,
        headers=_auth_headers(folder_id),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ошибка HTTP {e.code} при обращении к сервису: {err}") from e


def _parse_json_from_model_text(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Пустой ответ модели")
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


def fill_protocol_from_transcript(
    transcript: str,
    *,
    folder_id: Optional[str] = None,
    model_uri: Optional[str] = None,
) -> Dict[str, str]:
    """
    Один запрос к YandexGPT; модель возвращает JSON с четырьмя строковыми полями.
    """
    fid = (folder_id or resolve_yandex_folder_id() or "").strip()
    if not fid:
        raise ValueError("Не задан идентификатор каталога (YANDEX_FOLDER_ID)")
    uri = (model_uri or resolve_yandex_model_uri() or "").strip()
    if not uri:
        raise ValueError("Не удалось сформировать адрес модели; проверьте идентификатор каталога")

    user_text = (
        JSON_INSTRUCTION.strip()
        + "\n\nТранскрипт консультации:\n\n"
        + transcript.strip()
    )
    body = {
        "modelUri": uri,
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 8000,
        },
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user", "text": user_text},
        ],
    }
    resp = _post_yandex_completion(body, fid)
    try:
        alts = resp["result"]["alternatives"]
        raw_text = alts[0]["message"]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Неожиданный ответ сервиса: {resp!r}") from e

    data = _parse_json_from_model_text(raw_text)
    out: Dict[str, str] = {}
    for key in PROTOCOL_FIELD_KEYS:
        val = data.get(key)
        out[key] = val.strip() if isinstance(val, str) else ("" if val is None else str(val))
    return out


def format_protocol_editor_text(consultation_date: str, fields: Dict[str, str]) -> str:
    """
    Один блок для редактирования: заголовки с двоеточием и текст от модели / врача.
    """
    c = (fields.get("complaints") or "").strip()
    a = (fields.get("anamnesis") or "").strip()
    cl = (fields.get("conclusion") or "").strip()
    r = (fields.get("recommendations") or "").strip()
    return (
        f"Дата: {consultation_date.strip()}\n\n"
        f"Жалобы: {c}\n\n"
        f"Анамнез: {a}\n\n"
        f"Заключение: {cl}\n\n"
        f"Рекомендации: {r}"
    )


def parse_protocol_editor_text(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Разбор текста редактора в дату и четыре поля протокола.
    Заголовки — строки, начинающиеся с «Дата:», «Жалобы:» и т.д.
    """
    out: Dict[str, str] = {k: "" for k in PROTOCOL_FIELD_KEYS}
    date_str = ""
    order = [
        ("Дата", "date"),
        ("Жалобы", "complaints"),
        ("Анамнез", "anamnesis"),
        ("Заключение", "conclusion"),
        ("Рекомендации", "recommendations"),
    ]
    current: Optional[str] = None
    acc: list[str] = []

    def flush() -> None:
        nonlocal date_str, out, acc
        body = "\n".join(acc).strip()
        if current == "date":
            date_str = body
        elif current in out:
            out[current] = body
        acc = []

    for raw in (text or "").splitlines():
        line = raw.rstrip("\r")
        hit: Optional[str] = None
        rest = ""
        for label, key in order:
            pref = f"{label}:"
            if line.startswith(pref):
                hit = key
                rest = line[len(pref) :].lstrip()
                break
        if hit is not None:
            flush()
            current = hit
            acc = [rest] if rest else []
        elif current is not None:
            acc.append(line)
    flush()
    return date_str, out
