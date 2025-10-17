from __future__ import annotations

import os
import io
import time
from pathlib import Path

import pytest
import httpx


pytestmark = [
    pytest.mark.skipif(os.getenv("KAFKA_E2E", "0") != "1", reason="Set KAFKA_E2E=1 to run Kafka E2E tests"),
]


def _api_base() -> str:
    return os.getenv("API_BASE", "http://localhost:8080")


def _make_auth_header() -> dict[str, str]:
    # Prefer jwt_helper to align with API env; fallback to conftest, then manual
    try:
        from .jwt_helper import build_auth_header  # type: ignore
        return build_auth_header()
    except Exception:
        try:
            from .conftest import auth_header_e2e  # type: ignore
            return auth_header_e2e()  # type: ignore[misc]
        except Exception:
            import jwt
            payload = {os.getenv("JWT_ID_CLAIM", "id"): 1, "sub": os.getenv("JWT_ACCESS_SUBJECT", "AccessToken")}
            alg = os.getenv("JWT_ALGORITHM", "HS256")
            secret = os.getenv("JWT_SECRET", "devsecret")
            token = jwt.encode(payload, secret, algorithm=alg)
            header_name = os.getenv("JWT_TOKEN_HEADER", "Authorization")
            bearer = os.getenv("JWT_BEARER_PREFIX", "Bearer")
            return {header_name: f"{bearer} {token}"}


def _count_events(session_no: int) -> int:
    from app.api.domain.infra.db.session import ENGINE
    with ENGINE.begin() as conn:
        res = conn.exec_driver_sql(
            "SELECT COUNT(*) FROM analyzed_sound_events WHERE sleep_session_no = %s",
            (session_no,),
        )
        return int(list(res)[0][0])


def _load_sample_bytes_and_mime() -> tuple[bytes, str]:
    base = Path(__file__).resolve().parents[2] / "environment" / "data" / "sound"
    candidates = [
        (base / "snoring_60_120.webm", "audio/webm"),
        (base / "snoring_segment_001.webm", "audio/webm"),
        (base / "snoring_60_120.wav", "audio/wav"),
        (base / "snoring.wav", "audio/wav"),
    ]
    for p, mime in candidates:
        if p.exists():
            return p.read_bytes(), mime
    pytest.skip("No sound sample found under tests/environment/data/sound")
    return b"", "application/octet-stream"


def test_sound_api_upload_end_to_end():
    client = httpx.Client(base_url=_api_base(), timeout=30.0)
    headers = _make_auth_header()

    # create session
    r = client.post("/api/analysis/session", headers=headers)
    assert r.status_code == 200, r.text
    session_no = int(r.json()["result"]["sleep_session_no"]) if "result" in r.json() else int(r.json().get("sleep_session_no", 0))
    assert session_no > 0

    before = _count_events(session_no)

    # upload sound
    data, mime = _load_sample_bytes_and_mime()
    files = {"file_instance": ("sample", io.BytesIO(data), mime)}
    r2 = client.patch("/api/analysis/sound", headers=headers, files=files)
    assert r2.status_code == 200, r2.text

    # poll db up to 90s
    deadline = time.time() + 90
    backoff = 0.2
    while time.time() < deadline:
        after = _count_events(session_no)
        if after - before >= 1:
            break
        time.sleep(backoff)
        backoff = min(2.0, backoff * 1.5)

    assert _count_events(session_no) - before >= 1


