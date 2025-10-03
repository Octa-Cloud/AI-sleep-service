from __future__ import annotations
# 이 파일은 정상/불량 EDF 변형 파일들을 업로드했을 때 API 동작을 검증합니다.

import os
from typing import Iterable

import pytest
from fastapi.testclient import TestClient
import time
from sqlalchemy import select


def get_edf_dir() -> str:
    # 테스트 데이터가 위치한 edf 디렉터리 경로를 반환합니다.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "environment", "data", "edf"))


def load_edf_bytes(filename: str) -> bytes:
    # EDF 파일을 바이너리로 읽어 반환합니다.
    path = os.path.join(get_edf_dir(), filename)
    with open(path, "rb") as f:
        return f.read()


OK_EDF_FILES = [f"brainwave-10min-ok{i}.edf" for i in range(1, 9)]
BAD_EDF_FILES = [
    "brainwave-10min-damaged.edf",
    "brainwave-10min-channel-bad.edf",
    "brainwave-20min.edf",
]


@pytest.mark.parametrize("file_name", OK_EDF_FILES)
def test_brainwave_ok_variants(client: TestClient, auth_header: dict[str, str], file_name: str):
    session_resp = client.post("/api/sleep/session/", headers=auth_header)
    assert session_resp.status_code == 200, session_resp.text
    edf_bytes = load_edf_bytes(file_name)
    files = {"file_instance": (file_name, edf_bytes, "application/octet-stream")}
    data = {}
    resp = client.patch("/api/sleep/data/brainwave/", files=files, data=data, headers=auth_header)
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}, body={resp.text}"
    time.sleep(3)
    from app.api.domain.infra.db.session import SessionLocal
    from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
    with SessionLocal() as db:
        sleep_levels = db.execute(select(SleepLevel)).scalars().all()
        assert len(sleep_levels) >= 1


@pytest.mark.parametrize("file_name", BAD_EDF_FILES)
def test_brainwave_bad_variants(client: TestClient, auth_header: dict[str, str], file_name: str):
    session_resp = client.post("/api/sleep/session/", headers=auth_header)
    assert session_resp.status_code == 200, session_resp.text
    edf_bytes = load_edf_bytes(file_name)
    files = {"file_instance": (file_name, edf_bytes, "application/octet-stream")}
    data = {}
    resp = client.patch("/api/sleep/data/brainwave/", files=files, data=data, headers=auth_header)
    # Validation should fail 400/422; either is acceptable depending on parsing
    assert resp.status_code in (400, 422), f"expected 400/422, got {resp.status_code}, body={resp.text}"



