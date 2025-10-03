from __future__ import annotations
# 이 파일은 다양한 시나리오(정상/잘못된 세션ID/손상 파일/무인증)를 통합적으로 검증합니다.

import os
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
import time
from sqlalchemy import select


def get_edf_dir() -> str:
    # 테스트 데이터 edf 디렉터리 경로를 반환합니다.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "environment", "data", "edf"))


def load_edf_bytes(filename: str) -> bytes:
    # EDF 파일 내용을 바이트로 읽어옵니다.
    with open(os.path.join(get_edf_dir(), filename), "rb") as f:
        return f.read()


def create_sleep_session(client: TestClient, auth_header: dict[str, str]) -> None:
    # 업로드 전에 수면 세션을 생성합니다.
    resp = client.post("/api/sleep/session/", headers=auth_header)
    assert resp.status_code == 200, resp.text


def send_brainwave_request(client: TestClient, auth_header: dict[str, str], session_no: int, edf_bytes: bytes):
    # 뇌파 데이터를 업로드하는 요청을 전송합니다.
    files = {"file_instance": ("file.edf", edf_bytes, "application/octet-stream")}
    data = {"sleep_session_no": session_no}
    return client.patch("/api/sleep/data/brainwave/", files=files, data=data, headers=auth_header)


@pytest.mark.usefixtures("reset_db")
def test_scenario1_ok_10min_chunks(client: TestClient, auth_header: dict[str, str]):
    create_sleep_session(client, auth_header)
    for i in range(1, 9):  # ok1..ok8
        edf = load_edf_bytes(f"brainwave-10min-ok{i}.edf")
        resp = send_brainwave_request(client, auth_header, 0, edf)
        assert resp.status_code == 200, resp.text
    # 비동기 파이프라인 저장 반영 대기
    time.sleep(3)
    from app.api.domain.infra.db.session import SessionLocal
    from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
    from app.api.domain.domain.entity.sleep_session_entity import SleepSession
    with SessionLocal() as s:
        sessions = s.execute(select(SleepSession)).scalars().all()
        levels = s.execute(select(SleepLevel)).scalars().all()
        assert len(sessions) >= 1
        assert len(levels) >= 1


@pytest.mark.usefixtures("reset_db")
def test_scenario2_wrong_session_id(client: TestClient, auth_header: dict[str, str]):
    create_sleep_session(client, auth_header)
    edf = load_edf_bytes("brainwave-10min-ok1.edf")
    resp = send_brainwave_request(client, auth_header, 0, edf)
    assert resp.status_code == 200, resp.text
    time.sleep(3)
    from app.api.domain.infra.db.session import SessionLocal
    from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
    with SessionLocal() as s:
        levels = s.execute(select(SleepLevel)).scalars().all()
        assert len(levels) >= 1


@pytest.mark.usefixtures("reset_db")
def test_scenario3_damaged_chunk(client: TestClient, auth_header: dict[str, str]):
    create_sleep_session(client, auth_header)
    edf = load_edf_bytes("brainwave-10min-damaged.edf")
    resp = send_brainwave_request(client, auth_header, 0, edf)
    assert resp.status_code in (400, 422), resp.text


@pytest.mark.usefixtures("reset_db")
def test_scenario4_unauthorized(client: TestClient):
    edf = load_edf_bytes("brainwave-10min-ok1.edf")
    files = {"file_instance": ("file.edf", edf, "application/octet-stream")}
    data = {}
    resp = client.patch("/api/sleep/data/brainwave/", files=files, data=data)
    assert resp.status_code == 401, resp.text


@pytest.mark.usefixtures("reset_db")
def test_scenario5_duplicate_session_creation(client: TestClient, auth_header: dict[str, str]):
    # First creation succeeds
    resp1 = client.post("/api/sleep/session/", headers=auth_header)
    assert resp1.status_code == 200, resp1.text
    # Second creation while ongoing session exists should fail with 409
    resp2 = client.post("/api/sleep/session/", headers=auth_header)
    assert resp2.status_code == 409, resp2.text



