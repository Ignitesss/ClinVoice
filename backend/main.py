# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import auth
import clinvoice_db
from backend.routers import audio_ws, auth as auth_router
from backend.routers import consultations
from backend.settings import get_settings
from clinvoice_cache import apply_disk_cache_layout

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    apply_disk_cache_layout()
    db_path = auth.resolve_sqlite_path_cli()
    clinvoice_db.init_db(db_path)
    try:
        n = clinvoice_db.purge_expired(db_path)
        if n:
            log.info("purge_expired: удалено консультаций: %s", n)
    except Exception as e:
        log.warning("purge_expired: %s", e)
    app.state.db_path = db_path
    app.state.transcriber = None
    app.state.transcriber_lock = threading.Lock()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="ClinVoice API", lifespan=lifespan)
    origins = [o.strip() for o in get_settings().clinvoice_cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(auth_router.router, prefix="/api")
    app.include_router(consultations.router, prefix="/api")
    app.include_router(audio_ws.router, prefix="/ws")
    return app


app = create_app()
