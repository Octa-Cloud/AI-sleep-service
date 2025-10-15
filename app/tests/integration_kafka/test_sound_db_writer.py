from __future__ import annotations

import os
import asyncio
from datetime import datetime, timedelta, timezone

import pytest


pytestmark = [
    pytest.mark.asyncio,
]


def _ensure_session(session_no: int) -> None:
    from app.api.domain.infra.db.session import ENGINE
    with ENGINE.begin() as conn:
        conn.exec_driver_sql(
            """
            INSERT INTO sleep_sessions (sleep_session_no, user_no, created_at)
            VALUES (%s, %s, NOW())
            ON DUPLICATE KEY UPDATE user_no = VALUES(user_no)
            """,
            (session_no, 1),
        )


def _count_events(session_no: int) -> int:
    from app.api.domain.infra.db.session import ENGINE
    with ENGINE.begin() as conn:
        res = conn.exec_driver_sql(
            "SELECT COUNT(*) FROM analyzed_sound_events WHERE sleep_session_no = %s",
            (session_no,),
        )
        return int(list(res)[0][0])


async def test_sound_db_writer_persists_batch():
    try:
        from app.common.kafka.dto import sound_pb2 as pb  # type: ignore
    except Exception:
        pytest.skip("sound_pb2.py not generated; run scripts/gen_protos.py")
        return

    from aiokafka import AIOKafkaProducer

    brokers = os.getenv("KAFKA_BROKERS", "localhost:29092")
    topic = os.getenv("TOPIC_SOUND_PERSIST_REQUESTS", "sound.persist.requests")

    session_no = int(os.getenv("E2E_TEST_SESSION_NO", "777101"))
    _ensure_session(session_no)

    before = _count_events(session_no)

    now = datetime.now(timezone.utc)
    events = []
    for i in range(10):
        events.append(pb.SoundAnalyzedEvent(  # type: ignore[attr-defined]
            sleep_session_no=session_no,
            at_ms=int((now + timedelta(seconds=i)).timestamp() * 1000),
            event="SNORE",
            trace_id="e2e-sound-batch",
        ))
    msg = pb.SoundPersistRequest(events=events, trace_id="e2e-sound-batch")  # type: ignore[attr-defined]
    key = f"{session_no}:e2e-sound-batch".encode("utf-8")
    headers = [("trace_id", b"e2e-sound-batch"), ("session_no", str(session_no).encode("utf-8")), ("content-type", b"application/x-protobuf;msg=SoundPersistRequest")]

    producer = AIOKafkaProducer(bootstrap_servers=brokers)
    await producer.start()
    try:
        await producer.send_and_wait(topic, value=msg.SerializeToString(), key=key, headers=headers)
    finally:
        await producer.stop()

    # wait for consumer to process
    await asyncio.sleep(3)
    after = _count_events(session_no)

    assert after - before >= 10


