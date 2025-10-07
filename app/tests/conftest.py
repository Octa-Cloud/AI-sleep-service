from __future__ import annotations
# 이 파일은 테스트 공통 환경 설정과 픽스처(client, auth_header, reset_db 등)를 제공합니다.
# pytest 설정 파일

import os
import sys
import time
from pathlib import Path
from typing import Iterator

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import text as _sql_text

# Ensure project root is first on sys.path so imports like 'app.api...' resolve
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# Ensure ORM metadata has a minimal 'users' table for FK resolution during imports
try:
    from sqlalchemy import Table, Column, BigInteger  # type: ignore
    from app.api.domain.domain.entity.base import Base  # type: ignore
    if 'users' not in Base.metadata.tables:
        Table('users', Base.metadata, Column('user_no', BigInteger, primary_key=True))
except Exception:
    pass


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

    # Disable Kafka for standard tests; E2E Kafka tests opt-in via KAFKA_E2E=1
    os.environ.setdefault("KAFKA_ENABLED", "0")
    os.environ.setdefault("KAFKA_PROTOBUF_ENABLED", "1")
    # If running Kafka E2E, turn Kafka on
    if os.getenv("KAFKA_E2E", "0") == "1":
        os.environ["KAFKA_ENABLED"] = "1"


# Note: MySQL auto-start via docker compose was removed on purpose. Use the
# scripts under app/tests/environment/mysql/ to manage the DB lifecycle.

def _make_access_token(user_no: int = 1, exp_seconds: int = 3600) -> str:
    secret = os.getenv("JWT_SECRET", "test-secret-key")
    alg = os.getenv("JWT_ALGORITHM", "HS256")
    payload = {
        "sub": os.getenv("JWT_ACCESS_SUBJECT", "AccessToken"),
        os.getenv("JWT_ID_CLAIM", "id"): int(user_no),
        "exp": int(time.time()) + int(exp_seconds),
    }
    return jwt.encode(payload, secret, algorithm=alg)


def _setup_mysql() -> sessionmaker[Session]:
    from app.api.domain.infra.db import session as db_session

    user = os.getenv("DB_USER", "test")
    password = os.getenv("DB_PASSWORD", "testpw")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3307"))
    database = os.getenv("DB_NAME", "sleep_test")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"

    engine = create_engine(url, future=True)
    SessionLocal = sessionmaker(bind=engine, class_=Session, autocommit=False, autoflush=False, future=True)

    db_session.ENGINE = engine
    db_session.SessionLocal = SessionLocal  # type: ignore
    # Ensure decorator uses the test SessionLocal without touching app code
    try:
        from app.api.common.decorator import session_scope as _session_scope  # type: ignore
        _session_scope.SessionLocal = SessionLocal  # type: ignore[attr-defined]
    except Exception:
        pass
    return SessionLocal


@pytest.fixture(scope="session")
def test_session_factory() -> sessionmaker[Session]:
    _set_env()
    return _setup_mysql()


@pytest.fixture()
def client(test_session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    # 모델 파일 의존성을 제거하기 위해 테스트 시 TensorFlow/NumPy 로더를 스텁 처리합니다.
    try:
        import numpy as _np  # type: ignore
        import tensorflow as _tf  # type: ignore

        class _FakeModel:
            def predict(self, arr, verbose: int = 0):
                # 클래스 6개 가정, 모두 0 확률 반환
                batch = arr.shape[0] if hasattr(arr, 'shape') else 1
                return _np.zeros((batch, 6), dtype=_np.float32)

        _tf.keras.models.load_model = lambda *a, **k: _FakeModel()  # type: ignore[attr-defined]
        _np.load = lambda *a, **k: _np.ones((1,), dtype=_np.float32)  # type: ignore[assignment]
    except Exception:
        pass

    from app.api.main import app

    class _DummyUseCase:
        async def execute(self, sleep_session_no: int, edf_bytes: bytes) -> None:
            # Run real validator so bad inputs still fail fast
            container = getattr(app, "container", None) or app.state.container
            validator = getattr(container, "brainwave_validator")
            validator.validate(edf_bytes)
            # Simulate persistence by inserting a minimal SleepLevel row via service
            from datetime import datetime, timezone
            from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData
            sleeplevel_service = getattr(container, "brainwave_sleeplevel")
            entities = sleeplevel_service.data_to_entities(int(sleep_session_no), [
                SleepLevelData(level=1, recorded_at=datetime.now(timezone.utc))
            ])
            sleeplevel_service.insert_bulk(entities)
            return None

    container = getattr(app, "container", None) or app.state.container
    # Override brainwave use case factory only when not running Kafka E2E
    if os.getenv("KAFKA_E2E", "0") != "1":
        if hasattr(container, "brainwave_usecase_factory"):
            container.brainwave_usecase_factory = lambda: _DummyUseCase()

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_header() -> dict[str, str]:
    token = _make_access_token(user_no=1)
    header_name = os.getenv("JWT_TOKEN_HEADER", "Authorization")
    bearer = os.getenv("JWT_BEARER_PREFIX", "Bearer")
    return {header_name: f"{bearer} {token}"}


@pytest.fixture(autouse=True)
def reset_db(test_session_factory: sessionmaker[Session]) -> Iterator[None]:
    """Function-scoped: drop all tables and recreate from schema.sql, then seed base data."""
    _set_env()
    user = os.getenv("DB_USER", "test")
    password = os.getenv("DB_PASSWORD", "testpw")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3307"))
    database = os.getenv("DB_NAME", "sleep_test")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"

    engine = create_engine(url, future=True)
    # Wait briefly for MySQL readiness (in case tests start immediately)
    start = time.time()
    while True:
        try:
            with engine.connect() as conn:
                conn.execute(_sql_text("SELECT 1"))
            break
        except Exception:
            if time.time() - start > 30:
                raise
            time.sleep(1)
    # Drop and recreate from schema.sql only, do not rely on ORM models
    drop_sql = [
        "SET FOREIGN_KEY_CHECKS=0",
        "DROP TABLE IF EXISTS analysis_steps",
        "DROP TABLE IF EXISTS analysis_details",
        "DROP TABLE IF EXISTS sleep_time_details",
        "DROP TABLE IF EXISTS daily_reports",
        "DROP TABLE IF EXISTS score_prediction_points",
        "DROP TABLE IF EXISTS periodic_reports",
        "DROP TABLE IF EXISTS analyzed_sleep_levels",
        "DROP TABLE IF EXISTS analyzed_sound_events",
        "DROP TABLE IF EXISTS sleep_sessions",
        "DROP TABLE IF EXISTS users",
        "SET FOREIGN_KEY_CHECKS=1",
    ]
    with engine.begin() as conn:
        for stmt in drop_sql:
            conn.exec_driver_sql(stmt)
        # recreate from schema.sql
        schema_path = Path(__file__).resolve().parent / "environment" / "data" / "schema.sql"
        sql_text = schema_path.read_text(encoding="utf-8")
        # naive split by ';' safe enough for our schema
        for raw in sql_text.split(';'):
            stmt = raw.strip()
            if stmt:
                conn.exec_driver_sql(stmt)
        # seed base data: user 1
        conn.exec_driver_sql(
            "INSERT INTO users(user_no, name, nickname, email, password, gender) VALUES (1, 'User1', 'u1', 'user1@example.com', 'x', NULL)"
        )
    yield


