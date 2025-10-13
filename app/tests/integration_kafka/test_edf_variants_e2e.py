from __future__ import annotations

import os
import io
import time
from typing import Iterable

import pytest
import httpx
from sqlalchemy import select


pytestmark = [
    pytest.mark.skipif(os.getenv("KAFKA_E2E", "0") != "1", reason="Set KAFKA_E2E=1 to run Kafka E2E tests"),
]


def _api_base() -> str:
    return os.getenv("API_BASE", "http://localhost:8080")


def _edf_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "environment", "data", "edf"))


def _load_edf_bytes(filename: str) -> bytes:
    path = os.path.join(_edf_dir(), filename)
    if not os.path.exists(path):
        pytest.skip(f"EDF not found: {path}")
    with open(path, "rb") as f:
        return f.read()


def _count_levels() -> int:
    from app.api.domain.infra.db.session import ENGINE
    with ENGINE.begin() as conn:
        res = conn.exec_driver_sql("SELECT COUNT(*) FROM analyzed_sleep_levels")
        return int(list(res)[0][0])


OK_EDF_FILES: Iterable[str] = [f"brainwave-10min-ok{i}.edf" for i in range(1, 9)]
BAD_EDF_FILES: Iterable[str] = [
    "brainwave-10min-damaged.edf",
    "brainwave-10min-channel-bad.edf",
    "brainwave-20min.edf",
]


@pytest.mark.parametrize("file_name", OK_EDF_FILES)
def test_ok_edfs_return_200_and_persist(auth_header_e2e, file_name: str) -> None:
    client = httpx.Client(base_url=_api_base(), timeout=30.0)

    # create session
    r = client.post("/api/analysis/session", headers=auth_header_e2e)
    assert r.status_code == 200, r.text

    before = _count_levels()

    edf = _load_edf_bytes(file_name)
    files = {"file_instance": (file_name, io.BytesIO(edf), "application/octet-stream")}
    r2 = client.patch("/api/analysis/brainwave", headers=auth_header_e2e, files=files)
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


@pytest.mark.parametrize("file_name", BAD_EDF_FILES)
def test_bad_edfs_return_4xx(auth_header_e2e, file_name: str) -> None:
    client = httpx.Client(base_url=_api_base(), timeout=30.0)

    # create session
    r = client.post("/api/analysis/session", headers=auth_header_e2e)
    assert r.status_code == 200, r.text

    edf = _load_edf_bytes(file_name)
    files = {"file_instance": (file_name, io.BytesIO(edf), "application/octet-stream")}
    r2 = client.patch("/api/analysis/brainwave", headers=auth_header_e2e, files=files)
    assert r2.status_code in (400, 422), r2.text


