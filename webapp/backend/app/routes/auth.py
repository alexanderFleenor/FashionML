"""Single-password auth using signed session cookies (Starlette SessionMiddleware)."""

import bcrypt
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from ..config import settings

router = APIRouter()


class LoginBody(BaseModel):
    password: str


def _password_matches(submitted: str) -> bool:
    if settings.PASSWORD_HASH:
        try:
            return bcrypt.checkpw(submitted.encode("utf-8"), settings.PASSWORD_HASH.encode("utf-8"))
        except ValueError:
            return False
    if settings.PASSWORD:
        # Plain compare is enough for this local app.
        return submitted == settings.PASSWORD
    # Neither configured: refuse all logins rather than silently allowing entry.
    return False


@router.post("/login")
def login(body: LoginBody, request: Request):
    if not _password_matches(body.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid password")
    request.session["authed"] = True
    return {"ok": True}


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    if not request.session.get("authed"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return {"authed": True}


def require_auth(request: Request) -> None:
    """Dependency: raise 401 if the session is not authed."""
    if not request.session.get("authed"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
