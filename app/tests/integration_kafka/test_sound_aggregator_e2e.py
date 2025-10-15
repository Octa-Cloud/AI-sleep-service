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


async def test_sound_aggregator_produces_persist_and_dbwriter_saves():
    try:
        from app.common.kafka.dto import sound_pb2 as pb  # type: ignore
    except Exception:
        pytest.skip("sound_pb2.py not generated; run scripts/gen_protos.py")
        return

    from aiokafka import AIOKafkaProducer

    brokers = os.getenv("KAFKA_BROKERS", "localhost:29092")
    topic = os.getenv("TOPIC_SOUND_ANALYZED_EVENT", "sound.analyzed.event")

    session_no = int(os.getenv("E2E_TEST_SESSION_NO", "777201"))
    _ensure_session(session_no)

    before = _count_events(session_no)

    now = datetime.now(timezone.utc)
    end_index = 9
    producer = AIOKafkaProducer(bootstrap_servers=brokers)
    await producer.start()
    try:
        for i in range(end_index + 1):
            evt = pb.SoundAnalyzedEvent(  # type: ignore[attr-defined]
                sleep_session_no=int(session_no),
                at_ms=int((now + timedelta(milliseconds=i * 480)).timestamp() * 1000),
                event="SNORE",
                trace_id="e2e-sound-agg",
            )
            key = f"{session_no}:e2e-sound-agg".encode("utf-8")
            headers = [
                ("trace_id", b"e2e-sound-agg"),
                ("session_no", str(session_no).encode("utf-8")),
                ("epoch_index", str(i).encode("utf-8")),
                ("epoch_end_index", str(end_index).encode("utf-8")),
                ("content-type", b"application/x-protobuf;msg=SoundAnalyzedEvent"),
            ]
            await producer.send_and_wait(topic, value=evt.SerializeToString(), key=key, headers=headers)
    finally:
        await producer.stop()

    # Wait for aggregator+db-writer to persist
    deadline = datetime.now(timezone.utc) + timedelta(seconds=60)
    backoff = 0.2
    while datetime.now(timezone.utc) < deadline:
        after = _count_events(session_no)
        if after - before >= end_index + 1:
            break
        await asyncio.sleep(backoff)
        backoff = min(2.0, backoff * 1.5)

    assert _count_events(session_no) - before >= end_index + 1


