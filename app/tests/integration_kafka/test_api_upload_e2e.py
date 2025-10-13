from __future__ import annotations

import os
import io
import time

import pytest
import httpx


pytestmark = [
    pytest.mark.skipif(os.getenv("KAFKA_E2E", "0") != "1", reason="Set KAFKA_E2E=1 to run Kafka E2E tests"),
]


def _api_base() -> str:
    return os.getenv("API_BASE", "http://localhost:8080")


def _make_auth_header() -> dict[str, str]:
    # Prefer shared fixture-like behavior by reading .env via integration_kafka conftest
    try:
        from .conftest import auth_header_e2e  # type: ignore
        return auth_header_e2e()  # type: ignore[misc]
    except Exception:
        import jwt
        payload = {os.getenv("JWT_ID_CLAIM", "id"): 1, "sub": os.getenv("JWT_ACCESS_SUBJECT", "AccessToken")}
        token = jwt.encode(payload, os.getenv("JWT_SECRET", "devsecret"), algorithm=os.getenv("JWT_ALGORITHM", "HS256"))
        return {os.getenv("JWT_TOKEN_HEADER", "Authorization"): f"{os.getenv('JWT_BEARER_PREFIX', 'Bearer')} {token}"}


def _count_levels() -> int:
    from app.api.domain.infra.db.session import ENGINE
    with ENGINE.begin() as conn:
        res = conn.exec_driver_sql("SELECT COUNT(*) FROM analyzed_sleep_levels")
        return int(list(res)[0][0])


def _load_sample() -> bytes:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "environment", "data", "edf"))
    for name in ("brainwave-10min-ok1.edf", "brainwave-10min.edf"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
    pytest.skip("EDF sample not found")
    return b""


def test_api_upload_end_to_end(auth_header_e2e):
    client = httpx.Client(base_url=_api_base(), timeout=30.0)
    headers = auth_header_e2e

    # create session
    r = client.post("/api/analysis/session", headers=headers)
    assert r.status_code == 200, r.text

    before = _count_levels()

    # upload edf
    edf = _load_sample()
    files = {"file_instance": ("file.edf", io.BytesIO(edf), "application/octet-stream")}
    r2 = client.patch("/api/analysis/brainwave", headers=headers, files=files)
    assert r2.status_code == 200, r2.text

    # poll db up to 60s
    deadline = time.time() + 60
    backoff = 0.2
    while time.time() < deadline:
        after = _count_levels()
        if after - before >= 1:
            break
        time.sleep(backoff)
        backoff = min(2.0, backoff * 1.5)

    assert _count_levels() - before >= 1


