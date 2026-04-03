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
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_MAX_DIAG_MSG = 500

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


class _PostPreservingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """
    Google Apps Script /exec часто отвечает 302; стандартный клиент сбрасывает POST.
    Повторяем POST с тем же телом на Location.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        method = req.get_method()
        if method != "POST" or req.data is None:
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        if code not in (301, 302, 303, 307, 308):
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        new_headers = {}
        for k, v in req.header_items():
            lk = k.lower()
            if lk in ("host", "content-length", "connection"):
                continue
            new_headers[k] = v
        return urllib.request.Request(
            newurl,
            data=req.data,
            headers=new_headers,
            method="POST",
        )


def _sheets_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_PostPreservingRedirectHandler())


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


def metrics_sheets_secrets_configured() -> bool:
    """True, если заданы URL и секрет для отправки метрик."""
    u, s = _sheets_url_and_secret()
    return bool(u and s)


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


def _truncate(msg: str) -> str:
    msg = msg.replace("\n", " ").strip()
    if len(msg) > _MAX_DIAG_MSG:
        return msg[:_MAX_DIAG_MSG] + "…"
    return msg


def submit_metrics_row_to_sheets(row: List[Any]) -> Tuple[bool, Optional[str]]:
    """
    POST JSON { "secret": "...", "row": [...] } на Web App.
    Возвращает (успех, сообщение_об_ошибке_или_None).
    Сообщения без PHI (без row и без текстов консультаций).
    """
    url, secret = _sheets_url_and_secret()
    if not url or not secret:
        return False, "Не заданы CLINVOICE_SHEETS_WEBAPP_URL или CLINVOICE_SHEETS_SECRET"

    payload = json.dumps({"secret": secret, "row": row}, ensure_ascii=False)
    req = urllib.request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    opener = _sheets_opener()
    try:
        with opener.open(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                msg = f"HTTP {resp.status}: {_truncate(body)}"
                logger.warning("metrics_sheets: %s", msg)
                return False, msg
            try:
                data = json.loads(body)
                if not data.get("ok", True):
                    err = data.get("error", body)
                    msg = f"Ответ сервера ok:false: {_truncate(str(err))}"
                    logger.warning("metrics_sheets: %s", msg)
                    return False, msg
            except json.JSONDecodeError:
                pass
            return True, None
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        msg = f"HTTP {e.code}: {_truncate(raw)}"
        logger.warning("metrics_sheets: %s", msg)
        return False, msg
    except urllib.error.URLError as e:
        msg = f"Сеть: {e.reason!s}"
        logger.warning("metrics_sheets: URLError %s", e.reason)
        return False, msg
    except Exception as e:
        msg = f"{type(e).__name__}: {e!s}"
        logger.warning("metrics_sheets: %s", msg)
        return False, _truncate(msg)
