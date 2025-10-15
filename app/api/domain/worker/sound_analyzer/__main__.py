from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.common import config
from app.api.domain.application.service.sound.sound_analyzer_service import SoundAnalyzerService
from app.api.domain.domain.vo.chunked_data_value_object import SoundChunkData
from app.api.domain.domain.entity.analyzed_data_entity import SoundEventType


async def run() -> int:
    brokers = config.KAFKA_BROKERS
    in_topic = config.TOPIC_SOUND_SPLIT_EPOCHS
    out_topic = config.TOPIC_SOUND_ANALYZED_EVENT

    consumer = AIOKafkaConsumer(
        in_topic,
        bootstrap_servers=brokers,
        group_id=config.GROUP_SOUND_ANALYZER,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        session_timeout_ms=30000,
        heartbeat_interval_ms=3000,
        max_poll_interval_ms=300000,
    )
    producer = AIOKafkaProducer(bootstrap_servers=brokers)
    analyzer = SoundAnalyzerService()
    logger = logging.getLogger("sound.analyzer")

    try:
        await consumer.start()
        await producer.start()
        while True:
            try:
                msg = await consumer.getone()
                value = msg.value or b""
                hdrs_in = {k: (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)) for k, v in (msg.headers or [])}
                from app.common.kafka.dto import sound_pb2 as pb  # type: ignore
                split = pb.SoundSplitChunk()
                split.ParseFromString(value)
                logger.info("recv")

                # Base time: validation-complete timestamp + chunk start offset
                base_ms = int(hdrs_in.get("recorded_at_ms", "0") or 0)
                base = datetime.fromtimestamp(base_ms / 1000.0, tz=timezone.utc)
                chunk_start = base + timedelta(milliseconds=int(split.start_ms))
                chunk = SoundChunkData(data=split.data, start_at=chunk_start, sampling_rate_hz=float(split.sr))
                # Run heavy analysis outside the event loop to avoid heartbeat starvation
                events = await asyncio.to_thread(analyzer.analyze, [chunk])

                allowed_events = {e.value for e in SoundEventType}
                for ev in events:
                    evt_at_ms = int((ev.recorded_at).timestamp() * 1000)
                    evt = pb.SoundAnalyzedEvent(
                        sleep_session_no=int(split.session_no),
                        at_ms=evt_at_ms,
                        event=ev.event.value,
                        trace_id=split.trace_id,
                    )
                    headers = [
                        ("trace_id", split.trace_id.encode()),
                        ("session_no", str(int(split.session_no)).encode()),
                        ("content-type", b"application/x-protobuf;msg=SoundAnalyzedEvent"),
                        ("epoch_index", str(int(hdrs_in.get("epoch_index", "0"))).encode()),
                        ("epoch_end_index", str(int(hdrs_in.get("epoch_end_index", "0"))).encode()),
                    ]
                    # filter: only send events defined in entity enum
                    if evt.event not in allowed_events:
                        logger.info(
                            "analyzer_skip_unsupported_event",
                            extra={
                                "trace_id": split.trace_id,
                                "session_no": int(split.session_no),
                                "event": evt.event,
                            },
                        )
                        continue
                    out_bytes = evt.SerializeToString()
                    try:
                        await producer.send_and_wait(out_topic, key=split.trace_id.encode(), value=out_bytes, headers=headers)  # type: ignore[arg-type]
                        logger.info("sent")
                    except Exception as e:
                        logger.exception(
                            "analyzer_send_error",
                            extra={
                                "trace_id": split.trace_id,
                                "session_no": int(split.session_no),
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "error_repr": repr(e),
                                "topic": out_topic,
                                "payload_len": len(out_bytes),
                            },
                        )
                await consumer.commit()
            except asyncio.CancelledError:
                logger.info("analyzer_cancelled")
                raise
            except Exception:
                logger.exception("analyzer_loop_error")
                await asyncio.sleep(0.5)
    finally:
        logger.info("analyzer_stopping")
        try:
            await consumer.stop()
        finally:
            await producer.stop()
    return 0


def main() -> int:
    return asyncio.run(run())


