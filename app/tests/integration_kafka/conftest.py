from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest


def _load_root_env() -> None:
    root = Path(__file__).resolve().parents[3]
    env_path = root / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(env_path)
            return
        except Exception:
            pass
        # naive fallback: minimal parsing KEY=VALUE lines
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"')
                    os.environ.setdefault(k, v)
        except Exception:
            pass


@pytest.fixture(scope="session")
def auth_header_e2e() -> dict[str, str]:
    _load_root_env()
    try:
        import jwt  # type: ignore
    except Exception:
        pytest.skip("pyjwt not installed; cannot mint auth token")
        return {}

    secret = os.getenv("JWT_SECRET", "devsecret")
    alg = os.getenv("JWT_ALGORITHM", "HS256")
    sub = os.getenv("JWT_ACCESS_SUBJECT", "AccessToken")
    id_key = os.getenv("JWT_ID_CLAIM", "id")
    user_no = int(os.getenv("E2E_USER_NO", "1"))
    import time as _t
    payload = {"sub": sub, id_key: user_no, "exp": int(_t.time()) + 3600}
    token = jwt.encode(payload, secret, algorithm=alg)
    header_name = os.getenv("JWT_TOKEN_HEADER", "Authorization")
    bearer = os.getenv("JWT_BEARER_PREFIX", "Bearer")
    return {header_name: f"{bearer} {token}"}


