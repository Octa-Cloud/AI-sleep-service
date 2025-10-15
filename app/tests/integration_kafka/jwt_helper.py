from __future__ import annotations

import os
import time
from typing import Optional, Dict


def _strip_quotes(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("\"") and value.endswith("\""):
        return value.strip("\"")
    return value


def mint_token(user_no: int, exp_seconds: int, subject: Optional[str] = None) -> Optional[str]:
    try:
        import jwt  # type: ignore
    except Exception:
        return None

    secret = _strip_quotes(os.getenv("JWT_SECRET"))
    alg = os.getenv("JWT_ALGORITHM", "HS256")
    sub = subject or os.getenv("JWT_ACCESS_SUBJECT", "AccessToken")
    id_key = os.getenv("JWT_ID_CLAIM", "id")

    payload = {
        "sub": sub,
        id_key: int(user_no),
        "exp": int(time.time()) + int(exp_seconds),
    }
    return jwt.encode(payload, secret, algorithm=alg)


def build_auth_header(user_no: int = 1, exp_seconds: int = 3600, subject: Optional[str] = None) -> Dict[str, str]:
    token = mint_token(user_no, exp_seconds, subject) or ""
    header_name = os.getenv("JWT_TOKEN_HEADER", "Authorization")
    bearer = os.getenv("JWT_BEARER_PREFIX", "Bearer")
    return {header_name: f"{bearer} {token}"}
