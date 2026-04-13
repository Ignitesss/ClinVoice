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
from typing import Any, Dict, Optional

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
