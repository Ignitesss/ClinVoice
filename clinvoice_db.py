# -*- coding: utf-8 -*-
"""
SQLite persistence for ClinVoice: consultations + draft snapshots; TTL для старых записей.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Optional

DEFAULT_TTL_HOURS = 48
_SCHEMA_VERSION = 1


def resolve_ttl_hours() -> int:
    raw = (os.environ.get("CLINVOICE_DB_TTL_HOURS") or "").strip()
    if raw.isdigit():
        return max(1, min(8760, int(raw)))
    try:
        import streamlit as st

        if hasattr(st, "secrets") and st.secrets and "CLINVOICE_DB_TTL_HOURS" in st.secrets:
            v = str(st.secrets["CLINVOICE_DB_TTL_HOURS"]).strip()
            if v.isdigit():
                return max(1, min(8760, int(v)))
    except Exception:
        pass
    return DEFAULT_TTL_HOURS


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS consultations (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft'
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                content TEXT NOT NULL,
                saved_at INTEGER NOT NULL,
                FOREIGN KEY (consultation_id) REFERENCES consultations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_cid_type_saved
                ON artifacts (consultation_id, artifact_type, saved_at DESC);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if not row:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            conn.commit()


def consultation_exists(path: str, cid: str) -> bool:
    if not cid:
        return False
    with _connect(path) as conn:
        r = conn.execute(
            "SELECT 1 FROM consultations WHERE id = ? LIMIT 1", (cid,)
        ).fetchone()
        return r is not None


def upsert_consultation_row(path: str, cid: str, status: str = "draft") -> None:
    now = int(time.time())
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT created_at FROM consultations WHERE id = ?", (cid,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE consultations SET updated_at = ?, status = ? WHERE id = ?",
                (now, status, cid),
            )
        else:
            conn.execute(
                "INSERT INTO consultations (id, created_at, updated_at, status) VALUES (?,?,?,?)",
                (cid, now, now, status),
            )
        conn.commit()


def save_draft_snapshot(path: str, cid: str, payload: Dict[str, Any]) -> None:
    """Сохраняет JSON-снимок состояния UI (artifact_type=draft_snapshot)."""
    upsert_consultation_row(path, cid, str(payload.get("status") or "draft"))
    blob = json.dumps(payload, ensure_ascii=False)
    now = int(time.time())
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO artifacts (consultation_id, artifact_type, content, saved_at)
            VALUES (?, 'draft_snapshot', ?, ?)
            """,
            (cid, blob, now),
        )
        conn.commit()


def load_latest_snapshot(path: str, cid: str) -> Optional[Dict[str, Any]]:
    with _connect(path) as conn:
        row = conn.execute(
            """
            SELECT content FROM artifacts
            WHERE consultation_id = ? AND artifact_type = 'draft_snapshot'
            ORDER BY saved_at DESC LIMIT 1
            """,
            (cid,),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None


def purge_expired(path: str, ttl_hours: Optional[int] = None) -> int:
    """Удаляет консультации старше TTL по полю updated_at. Возвращает число удалённых строк."""
    ttl = ttl_hours if ttl_hours is not None else resolve_ttl_hours()
    cutoff = int(time.time()) - ttl * 3600
    with _connect(path) as conn:
        cur = conn.execute(
            "DELETE FROM consultations WHERE updated_at < ?", (cutoff,)
        )
        deleted = cur.rowcount
        conn.commit()
    return deleted
