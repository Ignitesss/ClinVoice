# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    clinvoice_jwt_secret: str = Field(
        default="",
        validation_alias="CLINVOICE_JWT_SECRET",
        description="Секрет подписи JWT (обязателен в проде).",
    )
    clinvoice_jwt_expire_hours: int = Field(default=48, validation_alias="CLINVOICE_JWT_EXPIRE_HOURS")
    clinvoice_cors_origins: str = Field(
        default="http://127.0.0.1:5173,http://localhost:5173",
        validation_alias="CLINVOICE_CORS_ORIGINS",
    )
    clinvoice_protocol_delay_sec: float = Field(
        default=10.0,
        validation_alias="CLINVOICE_PROTOCOL_DELAY_SEC",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_protocol_delay_sec() -> float:
    s = get_settings().clinvoice_protocol_delay_sec
    try:
        v = float(s)
    except ValueError:
        return 10.0
    return max(0.0, min(120.0, v))
