# -*- coding: utf-8 -*-
"""
SQLite persistence for ClinVoice: users, consultations (per user) + draft snapshots; TTL.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Optional

DEFAULT_TTL_HOURS = 48
_SCHEMA_VERSION = 2


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


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _migrate_consultations_add_user_id(conn: sqlite3.Connection) -> None:
    if "consultations" not in _table_names(conn):
        return
    cols = _table_columns(conn, "consultations")
    if "user_id" in cols:
        return
    conn.execute("ALTER TABLE consultations ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")
    u = conn.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
    if u:
        conn.execute(
            "UPDATE consultations SET user_id = ? WHERE user_id IS NULL",
            (u[0],),
        )


def init_db(path: str) -> None:
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        conn_ctx = _connect(path)
    except sqlite3.OperationalError as e:
        if "unable to open database file" in str(e).lower():
            raise sqlite3.OperationalError(
                f"{e}; path={abs_path!r} (проверьте CLINVOICE_SQLITE_PATH, "
                f"CLINVOICE_CACHE_DIR и права на запись в каталог)"
            ) from e
        raise
    with conn_ctx as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        tables = _table_names(conn)
        if "consultations" not in tables:
            conn.executescript(
                """
                CREATE TABLE consultations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_consultations_user_updated
                    ON consultations (user_id, updated_at);
                """
            )
        else:
            _migrate_consultations_add_user_id(conn)
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_consultations_user_updated
                    ON consultations (user_id, updated_at);
                """
            )

        conn.executescript(
            """
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
            """
        )
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if not row:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
        else:
            conn.execute(
                "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                (str(_SCHEMA_VERSION),),
            )
        conn.commit()


def create_user(path: str, username: str, password_plain: str) -> int:
    """Вставить пользователя с bcrypt-хешем. Вызывается из CLI скрипта."""
    import bcrypt

    u = (username or "").strip()
    if not u:
        raise ValueError("username required")
    pw = password_plain.encode("utf-8")
    if len(pw) < 1:
        raise ValueError("password required")
    h = bcrypt.hashpw(pw, bcrypt.gensalt())
    now = int(time.time())
    init_db(path)
    with _connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO users (username, password_hash, created_at)
            VALUES (?, ?, ?)
            """,
            (u, h, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_user_by_username(path: str, username: str) -> Optional[sqlite3.Row]:
    u = (username or "").strip()
    if not u:
        return None
    with _connect(path) as conn:
        return conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ? LIMIT 1",
            (u,),
        ).fetchone()


def consultation_exists(path: str, cid: str, user_id: int) -> bool:
    if not cid:
        return False
    with _connect(path) as conn:
        r = conn.execute(
            "SELECT 1 FROM consultations WHERE id = ? AND user_id = ? LIMIT 1",
            (cid, user_id),
        ).fetchone()
        return r is not None


def upsert_consultation_row(path: str, cid: str, user_id: int, status: str = "draft") -> None:
    now = int(time.time())
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT created_at FROM consultations WHERE id = ? AND user_id = ?",
            (cid, user_id),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE consultations SET updated_at = ?, status = ?
                WHERE id = ? AND user_id = ?
                """,
                (now, status, cid, user_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO consultations (id, user_id, created_at, updated_at, status)
                VALUES (?,?,?,?,?)
                """,
                (cid, user_id, now, now, status),
            )
        conn.commit()


def save_draft_snapshot(path: str, cid: str, user_id: int, payload: Dict[str, Any]) -> None:
    """Сохраняет JSON-снимок (artifact_type=draft_snapshot). Только для своей консультации."""
    upsert_consultation_row(path, cid, user_id, str(payload.get("status") or "draft"))
    blob = json.dumps(payload, ensure_ascii=False)
    now = int(time.time())
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO artifacts (consultation_id, artifact_type, content, saved_at)
            SELECT ?, 'draft_snapshot', ?, ?
            WHERE EXISTS (
                SELECT 1 FROM consultations WHERE id = ? AND user_id = ?
            )
            """,
            (cid, blob, now, cid, user_id),
        )
        conn.commit()


def load_latest_snapshot(path: str, cid: str, user_id: int) -> Optional[Dict[str, Any]]:
    with _connect(path) as conn:
        row = conn.execute(
            """
            SELECT a.content FROM artifacts a
            INNER JOIN consultations c ON c.id = a.consultation_id
            WHERE a.consultation_id = ?
              AND c.user_id = ?
              AND a.artifact_type = 'draft_snapshot'
            ORDER BY a.saved_at DESC
            LIMIT 1
            """,
            (cid, user_id),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None


def purge_expired(path: str, ttl_hours: Optional[int] = None) -> int:
    """Удаляет консультации старше TTL по updated_at (глобально по всем пользователям)."""
    ttl = ttl_hours if ttl_hours is not None else resolve_ttl_hours()
    cutoff = int(time.time()) - ttl * 3600
    with _connect(path) as conn:
        cur = conn.execute(
            "DELETE FROM consultations WHERE updated_at < ?", (cutoff,)
        )
        deleted = cur.rowcount
        conn.commit()
    return deleted
