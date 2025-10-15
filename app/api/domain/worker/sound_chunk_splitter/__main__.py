from __future__ import annotations

import asyncio
import logging
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.common import config
from app.api.domain.application.service.sound.sound_chunk_splitter_service import SoundChunkSplitterService


def _to_ms(delta_seconds: float) -> int:
    return int(round(delta_seconds * 1000))


async def run() -> int:
    brokers = config.KAFKA_BROKERS
    in_topic = config.TOPIC_SOUND_INPUT_RAW
    out_topic = config.TOPIC_SOUND_SPLIT_EPOCHS

    consumer = AIOKafkaConsumer(
        in_topic,
        bootstrap_servers=brokers,
        group_id=config.GROUP_SOUND_SPLITTER,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        session_timeout_ms=30000,
        heartbeat_interval_ms=3000,
        max_poll_interval_ms=300000,
    )
    producer = AIOKafkaProducer(bootstrap_servers=brokers)
    splitter = SoundChunkSplitterService()
    logger = logging.getLogger("sound.splitter")

    try:
        await consumer.start()
        await producer.start()
        while True:
            msg = await consumer.getone()
            value = msg.value or b""
            from app.common.kafka.dto import sound_pb2 as pb  # type: ignore
            hdrs_in = {k: (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)) for k, v in (msg.headers or [])}
            try:
                raw = pb.SoundInputRaw()
                raw.ParseFromString(value)
                logger.debug(f"splitter_recv trace_id={raw.trace_id} session_no={int(raw.session_no)} size={len(value)}")
                chunks = splitter.split(raw.data)
            except Exception as e:
                logger.error("splitter_parse_or_split_error", extra={"error": str(e)})
                chunks = []

            if chunks:
                base_start = chunks[0].start_at
                total = len(chunks)
                for idx, c in enumerate(chunks):
                    start_ms = _to_ms((c.start_at - base_start).total_seconds())
                    chunk_msg = pb.SoundSplitChunk(
                        data=c.data,
                        sr=float(c.sampling_rate_hz),
                        start_ms=start_ms,
                        trace_id=raw.trace_id,
                        session_no=raw.session_no,
                        recorded_at_ms=int(hdrs_in.get("recorded_at_ms", "0") or 0),
                    )
                    headers = [
                        ("trace_id", raw.trace_id.encode()),
                        ("session_no", str(int(raw.session_no)).encode()),
                        ("epoch_index", str(int(idx)).encode()),
                        ("epoch_end_index", str(int(total - 1)).encode()),
                        ("recorded_at_ms", hdrs_in.get("recorded_at_ms", "").encode()),
                        ("content-type", b"application/x-protobuf;msg=SoundSplitChunk"),
                    ]
                    try:
                        await producer.send_and_wait(out_topic, key=raw.trace_id.encode(), value=chunk_msg.SerializeToString(), headers=headers)  # type: ignore[arg-type]
                    except Exception:
                        logger.exception("splitter_send_error", extra={"trace_id": raw.trace_id, "session_no": int(raw.session_no)})
                    logger.debug(f"splitter_sent trace_id={raw.trace_id} session_no={int(raw.session_no)} start_ms={int(start_ms)} size={len(chunk_msg.data)}")
                logger.debug(f"splitter_sent_summary trace_id={raw.trace_id} session_no={int(raw.session_no)} epochs={len(chunks)}")
            else:
                logger.debug(f"splitter_no_chunks trace_id={getattr(raw, 'trace_id', '-')} session_no={int(getattr(raw, 'session_no', 0))}")
            await consumer.commit()
    except Exception as e:
        logger = logging.getLogger("sound.splitter")
        logger.error("splitter_fatal", extra={"error": str(e)})
    finally:
        try:
            await consumer.stop()
        finally:
            await producer.stop()
    return 0


def main() -> int:
    return asyncio.run(run())


