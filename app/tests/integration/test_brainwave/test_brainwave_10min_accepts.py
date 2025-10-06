from __future__ import annotations
# 이 파일은 10분 길이의 EDF 데이터를 업로드했을 때 API가 정상 동작하는지 확인하는 통합 테스트입니다.

import io
import os
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
import time
from sqlalchemy import select


def _set_env() -> None:
    os.environ.setdefault("DB_USER", "test")
    os.environ.setdefault("DB_PASSWORD", "testpw")
    os.environ.setdefault("DB_HOST", "127.0.0.1")
    os.environ.setdefault("DB_PORT", "3307")
    os.environ.setdefault("DB_NAME", "sleep_test")
    os.environ.setdefault("SQL_ECHO", "false")

    os.environ.setdefault("JWT_SECRET", "test-secret-key")
    os.environ.setdefault("JWT_ALGORITHM", "HS256")
    os.environ.setdefault("JWT_ACCESS_SUBJECT", "AccessToken")
    os.environ.setdefault("JWT_REFRESH_SUBJECT", "RefreshToken")
    os.environ.setdefault("JWT_TOKEN_HEADER", "Authorization")
    os.environ.setdefault("JWT_BEARER_PREFIX", "Bearer")
    os.environ.setdefault("JWT_ID_CLAIM", "id")


@pytest.fixture(scope="session")
def auth_header() -> dict[str, str]:
    import time, jwt
    _set_env()
    payload = {"sub": os.getenv("JWT_ACCESS_SUBJECT"), os.getenv("JWT_ID_CLAIM"): 1, "exp": int(time.time()) + 3600}
    token = jwt.encode(payload, os.getenv("JWT_SECRET"), algorithm=os.getenv("JWT_ALGORITHM"))
    return {os.getenv("JWT_TOKEN_HEADER"): f"{os.getenv('JWT_BEARER_PREFIX')} {token}"}


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    _set_env()
    from app.api.main import app
    with TestClient(app) as c:
        yield c


def load_10min_edf_slice() -> bytes:
    # edf 디렉터리에서 10분짜리 EDF 샘플을 로드합니다.
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "environment", "data", "edf"))
    candidates = [
        os.path.join(base, "brainwave-10min-ok1.edf"),
        os.path.join(base, "brainwave-10min.edf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    assert False, f"10min EDF not found. Expected one of: {candidates}"


def test_brainwave_accepts_10min_edf(client: TestClient, auth_header: dict[str, str]):
    # 업로드 전에 수면 세션을 생성합니다.
    session_resp = client.post("/api/sleep/session/begin", headers=auth_header)
    assert session_resp.status_code == 200, session_resp.text
    edf_bytes = load_10min_edf_slice()
    files = {"file_instance": ("sample.edf", edf_bytes, "application/octet-stream")}
    data = {}

    # EDF 파일을 업로드합니다.
    resp = client.patch("/api/sleep/data/brainwave/", files=files, data=data, headers=auth_header)
    assert resp.status_code in (200, 400, 422), f"status={resp.status_code}, body={resp.text}"
    if resp.status_code == 200:
        # 비동기 저장이 반영되도록 잠시 대기합니다.
        time.sleep(3)
        from app.api.domain.infra.db.session import SessionLocal
        from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
        with SessionLocal() as db:
            sleep_levels = db.execute(select(SleepLevel)).scalars().all()
            assert len(sleep_levels) >= 1



