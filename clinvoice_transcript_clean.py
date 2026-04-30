# -*- coding: utf-8 -*-
"""Удаление типичных «телевизионных» галлюцинаций Whisper (субтитры, титры и т.п.)."""

from __future__ import annotations

import re
from typing import Pattern

_TV_CAPTION_PATTERNS: tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"субтитры\s+сделаны\s+субтитрами\.?",
        r"субтитры\s+созданы\s+субтитрами\.?",
        r"субтитры\s+делали\s+субтитрами\.?",
        r"продолжение\s+следует\.?\.?",
        r"thanks?\s+for\s+watching\.?",
        r"subtitles?\s+by\s+\w+",
    )
)


def strip_whisper_tv_caption_artifacts(text: str) -> str:
    """Убирает известные шаблоны; схлопывает лишние пробелы."""
    if not (text or "").strip():
        return (text or "").strip()
    t = text.strip()
    for rx in _TV_CAPTION_PATTERNS:
        t = rx.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
