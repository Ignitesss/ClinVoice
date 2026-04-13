# -*- coding: utf-8 -*-
"""
Протокол консультации через OpenAI (structured outputs).

Рекомендуемая схема CSV для будущей разметки / обучения (MVP не парсит файлы):
  audio_path, category, transcript_ref, complaints, anamnesis, conclusion,
  recommendations, consultation_date_utc (опционально)

Тип задачи: генеративное структурирование (не дословное извлечение span).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

# Поля JSON = ключи словаря протокола (совпадают с OpenAI schema)
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


def _response_json_schema() -> Dict[str, Any]:
    props = {
        "complaints": {
            "type": "string",
            "description": "Жалобы и симптомы пациента",
        },
        "anamnesis": {
            "type": "string",
            "description": "Анамнез и сопутствующие сведения",
        },
        "conclusion": {
            "type": "string",
            "description": "Заключение врача",
        },
        "recommendations": {
            "type": "string",
            "description": "Рекомендации и назначения",
        },
    }
    return {
        "name": "therapist_consultation_protocol",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": props,
            "required": list(PROTOCOL_FIELD_KEYS),
            "additionalProperties": False,
        },
    }


def resolve_openai_api_key() -> Optional[str]:
    k = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if k:
        return k
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return None


def resolve_openai_model() -> str:
    m = (os.environ.get("OPENAI_MODEL") or "").strip()
    if m:
        return m
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "OPENAI_MODEL" in st.secrets:
            return str(st.secrets["OPENAI_MODEL"]).strip()
    except Exception:
        pass
    return "gpt-4o"


def fill_protocol_from_transcript(transcript: str, api_key: str, model: Optional[str] = None) -> Dict[str, str]:
    """
    Один вызов Chat Completions с structured output (json_schema).
    Возвращает словарь с ключами complaints, anamnesis, conclusion, recommendations.
    """
    from openai import OpenAI

    model_id = model or resolve_openai_model()
    client = OpenAI(api_key=api_key)
    user_content = (
        "Транскрипт консультации:\n\n" + transcript.strip() + "\n\n"
        "Заполни поля протокола согласно схеме."
    )
    fmt = {
        "type": "json_schema",
        "json_schema": _response_json_schema(),
    }
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=fmt,  # type: ignore[arg-type]
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise ValueError("Пустой ответ модели")
    data = json.loads(raw)
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
