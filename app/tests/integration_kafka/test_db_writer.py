from __future__ import annotations

import os
import asyncio
from datetime import datetime, timedelta, timezone

import pytest


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("KAFKA_E2E", "0") != "1", reason="Set KAFKA_E2E=1 to run Kafka E2E tests"),
]


async def _send_persist_request(session_no: int) -> None:
    try:
        from app.common.kafka.dto import brainwave_pb2 as pb  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip("brainwave_pb2.py not generated; run scripts/gen_protos.sh")
        return

    from aiokafka import AIOKafkaProducer

    brokers = os.getenv("KAFKA_BROKERS", "localhost:29092")
    topic = os.getenv("TOPIC_BRAINWAVE_PERSIST_REQUESTS", "brainwave.persist.requests")

    now = datetime.now(timezone.utc)
    items = []
    for i in range(20):
        items.append(pb.BrainwavePersistRequest.LevelItem(  # type: ignore[attr-defined]
            epoch_index=i,
            level=(i % 5),
            recorded_at_ms=int((now + timedelta(seconds=i * 30)).timestamp() * 1000),
        ))
    msg = pb.BrainwavePersistRequest(  # type: ignore[attr-defined]
        trace_id="e2e-test",
        session_no=int(session_no),
        levels=items,
    )
    key = f"{session_no}:e2e-test".encode("utf-8")
    headers = [("trace_id", b"e2e-test"), ("session_no", str(session_no).encode("utf-8")), ("content-type", b"application/x-protobuf;msg=BrainwavePersistRequest")]

    producer = AIOKafkaProducer(bootstrap_servers=brokers)
    await producer.start()
    try:
        await producer.send_and_wait(topic, value=msg.SerializeToString(), key=key, headers=headers)
    finally:
        await producer.stop()


def _ensure_session(session_no: int) -> None:
    # Ensure sleep_session exists
    from app.api.domain.infra.db.session import ENGINE
    with ENGINE.begin() as conn:
        # Minimal insert; assumes users table exists (tests/env mysql schema)
        # If session exists, ignore error using MySQL INSERT IGNORE pattern
        # SQLAlchemy Core upsert would be more involved; keep simple here.
        conn.exec_driver_sql(
            """
            INSERT INTO sleep_sessions (sleep_session_no, user_no, created_at)
            VALUES (%s, %s, NOW())
            ON DUPLICATE KEY UPDATE user_no = VALUES(user_no)
            """,
            (session_no, 1),
        )


def _count_levels(session_no: int) -> int:
    from app.api.domain.infra.db.session import ENGINE
    with ENGINE.begin() as conn:
        res = conn.exec_driver_sql(
            "SELECT COUNT(*) FROM analyzed_sleep_levels WHERE sleep_session_no = %s",
            (session_no,),
        )
        return int(list(res)[0][0])


async def test_db_writer_persists_levels():
    session_no = int(os.getenv("E2E_TEST_SESSION_NO", "999999"))
    _ensure_session(session_no)

    before = _count_levels(session_no)
    await _send_persist_request(session_no)

    # wait for consumer to process
    await asyncio.sleep(3)
    after = _count_levels(session_no)

    assert after - before >= 20


