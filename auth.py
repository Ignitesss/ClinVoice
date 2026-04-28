# -*- coding: utf-8 -*-
"""Проверка логина/пароля против SQLite users (bcrypt)."""

from __future__ import annotations

import os
from typing import Optional

import bcrypt

import clinvoice_db


def resolve_sqlite_path_cli() -> str:
    """Путь к БД без Streamlit (env CLINVOICE_SQLITE_PATH или ~/.cache/clinvoice/clinvoice.db)."""
    p = (os.environ.get("CLINVOICE_SQLITE_PATH") or "").strip()
    if not p:
        p = os.path.join(os.path.expanduser("~"), ".cache", "clinvoice", "clinvoice.db")
    parent = os.path.dirname(os.path.abspath(p))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return p


def verify_user(db_path: str, username: str, password: str) -> Optional[int]:
    """
    Возвращает user id при успехе, иначе None.
    """
    row = clinvoice_db.get_user_by_username(db_path, username)
    if not row:
        return None
    stored = row["password_hash"]
    if isinstance(stored, memoryview):
        stored = bytes(stored)
    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), stored)
    except ValueError:
        return None
    if not ok:
        return None
    return int(row["id"])
