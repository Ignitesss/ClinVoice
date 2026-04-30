#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python3 -m uvicorn backend.main:app --host "${CLINVOICE_BIND:-127.0.0.1}" --port "${CLINVOICE_PORT:-8000}" --workers 1
