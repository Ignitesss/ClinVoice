# -*- coding: utf-8 -*-
"""
Отправка обезличенных метрик в Google Sheets через Apps Script Web App.

Первая строка таблицы (заголовки, тот же порядок):
submitted_at, record_id, model_id, wer, wer_percent,
rouge1_f, rouge1_p, rouge1_r, rouge2_f, rouge2_p, rouge2_r,
rougeL_f, rougeL_p, rougeL_r
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Порядок колонок = порядок значений в row
METRICS_SHEET_HEADERS: tuple[str, ...] = (
    "submitted_at",
    "record_id",
    "model_id",
    "wer",
    "wer_percent",
    "rouge1_f",
    "rouge1_p",
    "rouge1_r",
    "rouge2_f",
    "rouge2_p",
    "rouge2_r",
    "rougeL_f",
    "rougeL_p",
    "rougeL_r",
)


def _sheets_url_and_secret() -> tuple[Optional[str], Optional[str]]:
    url = (os.environ.get("CLINVOICE_SHEETS_WEBAPP_URL") or "").strip()
    secret = (os.environ.get("CLINVOICE_SHEETS_SECRET") or "").strip()
    if not url or not secret:
        try:
            if hasattr(st, "secrets"):
                if not url and "CLINVOICE_SHEETS_WEBAPP_URL" in st.secrets:
                    url = str(st.secrets["CLINVOICE_SHEETS_WEBAPP_URL"]).strip()
                if not secret and "CLINVOICE_SHEETS_SECRET" in st.secrets:
                    secret = str(st.secrets["CLINVOICE_SHEETS_SECRET"]).strip()
        except Exception:
            pass
    return (url or None, secret or None)


def build_metrics_row(
    wer_results: Dict[str, Any],
    rouge_results: Optional[Dict[str, Dict[str, float]]],
    model_id: str,
) -> List[Any]:
    """Собирает строку для appendRow: только скаляры, без текстов консультаций."""
    submitted_at = datetime.now(timezone.utc).isoformat()
    record_id = str(uuid.uuid4())
    wer = float(wer_results["wer"])
    wer_percent = float(wer_results["wer_percentage"])

    def rouge_triplet(key: str) -> tuple[Any, Any, Any]:
        if not rouge_results or key not in rouge_results:
            return ("", "", "")
        r = rouge_results[key]
        return (float(r["f"]), float(r["p"]), float(r["r"]))

    r1f, r1p, r1r = rouge_triplet("rouge-1")
    r2f, r2p, r2r = rouge_triplet("rouge-2")
    rlf, rlp, rlr = rouge_triplet("rouge-l")

    return [
        submitted_at,
        record_id,
        str(model_id),
        wer,
        wer_percent,
        r1f,
        r1p,
        r1r,
        r2f,
        r2p,
        r2r,
        rlf,
        rlp,
        rlr,
    ]


def submit_metrics_row_to_sheets(row: List[Any]) -> bool:
    """
    POST JSON { "secret": "...", "row": [...] } на Web App.
    Возвращает True при HTTP 200 и теле с ok:true (если есть), иначе False.
    """
    url, secret = _sheets_url_and_secret()
    if not url or not secret:
        return False

    payload = json.dumps({"secret": secret, "row": row}, ensure_ascii=False)
    req = urllib.request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                logger.warning("metrics_sheets: HTTP %s", resp.status)
                return False
            try:
                data = json.loads(body)
                return bool(data.get("ok", True))
            except json.JSONDecodeError:
                return True
    except urllib.error.HTTPError as e:
        logger.warning("metrics_sheets: HTTPError %s", e.code)
        return False
    except urllib.error.URLError as e:
        logger.warning("metrics_sheets: URLError %s", e.reason)
        return False
    except Exception as e:
        logger.warning("metrics_sheets: %s", type(e).__name__)
        return False
