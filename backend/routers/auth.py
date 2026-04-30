# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

import auth
import clinvoice_db
from backend.deps import COOKIE_NAME, create_access_token, get_current_user_id, get_db_path

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=500)


@router.post("/login")
def login(body: LoginBody, response: Response, db_path: str = Depends(get_db_path)) -> dict:
    uid = auth.verify_user(db_path, body.username, body.password)
    if uid is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный логин или пароль")
    token = create_access_token(user_id=uid, username=body.username.strip())
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400 * 30,
        path="/",
    )
    return {"ok": True, "username": body.username.strip()}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user_id: int = Depends(get_current_user_id), db_path: str = Depends(get_db_path)) -> dict:
    row = clinvoice_db.get_user_by_id(db_path, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return {"id": int(row["id"]), "username": str(row["username"])}
