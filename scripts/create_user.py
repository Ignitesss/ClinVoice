#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Одноразовое создание пользователя в локальной БД ClinVoice.

Пример:
  export CLINVOICE_SQLITE_PATH=/path/to/clinvoice.db
  python scripts/create_user.py myuser

Пароль запросится интерактивно (getpass). Либо для CI:
  CLINVOICE_BOOTSTRAP_PASSWORD=secret python scripts/create_user.py myuser
"""

from __future__ import annotations

import argparse
import getpass
import os
import sqlite3
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import clinvoice_db  # noqa: E402
from auth import resolve_sqlite_path_cli  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Создать пользователя ClinVoice (bcrypt в SQLite).")
    p.add_argument("username", help="Логин")
    p.add_argument(
        "--db",
        metavar="PATH",
        help="Путь к SQLite (иначе CLINVOICE_SQLITE_PATH или ~/.cache/clinvoice/clinvoice.db)",
    )
    args = p.parse_args()

    db_path = (args.db or "").strip() or resolve_sqlite_path_cli()
    pw_env = (os.environ.get("CLINVOICE_BOOTSTRAP_PASSWORD") or "").strip()
    if pw_env:
        password = pw_env
    else:
        password = getpass.getpass("Пароль: ")
        confirm = getpass.getpass("Пароль ещё раз: ")
        if password != confirm:
            print("Пароли не совпадают.", file=sys.stderr)
            return 1

    clinvoice_db.init_db(db_path)
    try:
        uid = clinvoice_db.create_user(db_path, args.username, password)
    except sqlite3.IntegrityError as e:
        print(f"Ошибка уникальности (логин занят?): {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"Пользователь создан: id={uid}, username={args.username!r}, db={db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
