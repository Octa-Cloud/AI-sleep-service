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


def _count_levels(session_no: int) -> int:
    from app.api.domain.infra.db.session import ENGINE
    with ENGINE.begin() as conn:
        res = conn.exec_driver_sql(
            "SELECT COUNT(*) FROM analyzed_sleep_levels WHERE sleep_session_no = %s",
            (session_no,),
        )
        return int(list(res)[0][0])


async def test_aggregator_produces_persist_and_dbwriter_saves():
    try:
        from app.common.kafka.dto import brainwave_pb2 as pb  # type: ignore
    except Exception:
        pytest.skip("brainwave_pb2.py not generated; run scripts/gen_protos.sh")
        return

    from aiokafka import AIOKafkaProducer

    brokers = os.getenv("KAFKA_BROKERS", "localhost:29092")
    topic = os.getenv("TOPIC_BRAINWAVE_ANALYZED_EPOCH", "brainwave.analyzed.epoch")

    session_no = int(os.getenv("E2E_TEST_SESSION_NO", "777001"))
    _ensure_session(session_no)

    before = _count_levels(session_no)

    now = datetime.now(timezone.utc)
    end_index = 19
    producer = AIOKafkaProducer(bootstrap_servers=brokers)
    await producer.start()
    try:
        for i in range(end_index + 1):
            msg = pb.BrainwaveAnalyzedEpoch(  # type: ignore[attr-defined]
                trace_id="e2e-agg",
                session_no=int(session_no),
                epoch_index=i,
                epoch_end_index=end_index,
                level=(i % 5),
                recorded_at_ms=int((now + timedelta(seconds=i * 30)).timestamp() * 1000),
            )
            key = f"{session_no}:e2e-agg".encode("utf-8")
            headers = [
                ("trace_id", b"e2e-agg"),
                ("session_no", str(session_no).encode("utf-8")),
                ("content-type", b"application/x-protobuf;msg=BrainwaveAnalyzedEpoch"),
            ]
            await producer.send_and_wait(topic, value=msg.SerializeToString(), key=key, headers=headers)
    finally:
        await producer.stop()

    # Wait for aggregator+db-writer to persist
    deadline = datetime.now(timezone.utc) + timedelta(seconds=60)
    backoff = 0.2
    while datetime.now(timezone.utc) < deadline:
        after = _count_levels(session_no)
        if after - before >= end_index + 1:
            break
        await asyncio.sleep(backoff)
        backoff = min(2.0, backoff * 1.5)

    assert _count_levels(session_no) - before >= end_index + 1


