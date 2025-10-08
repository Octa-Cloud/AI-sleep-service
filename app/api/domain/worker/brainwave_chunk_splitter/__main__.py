from __future__ import annotations

import asyncio
import json
import os
from datetime import timedelta
from typing import Optional

import numpy as np
import logging
from aiokafka import AIOKafkaConsumer
from app.common import config

try:
    # Compiled via protoc to app/common/kafka/dto/brainwave_pb2.py
    from app.common.kafka.dto import brainwave_pb2 as pb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pb = None  # type: ignore

from app.common.kafka.producer import KafkaProducerClient
from app.api.domain.worker.common.runner import KafkaStageRunner
from app.api.domain.application.service.brainwave.brainwave_chunk_splitter_service import BrainwaveChunkSplitterService


async def run() -> int:
    brokers = config.KAFKA_BROKERS
    topic_in = config.TOPIC_BRAINWAVE_INPUT_RAW
    topic_out = config.TOPIC_BRAINWAVE_SPLIT_EPOCHS
    group_id = config.GROUP_BRAINWAVE_SPLITTER
    use_proto = True
    if pb is None:
        raise RuntimeError("Protobuf stubs not generated. Run scripts/gen_protos.sh before starting the worker.")

    consumer = AIOKafkaConsumer(
        topic_in,
        bootstrap_servers=brokers,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    producer = KafkaProducerClient(brokers)

    await producer.start()
    splitter = BrainwaveChunkSplitterService()
    logger = logging.getLogger("brainwave.Splitter")
    logger.setLevel(logging.INFO)
    async def _handle(value: bytes) -> None:
        edf_bytes: Optional[bytes] = None
        trace_id: str
        session_no: int
        epoch_sec: int

        obj = pb.BrainwaveInputRaw()  # type: ignore[attr-defined]
        obj.ParseFromString(value)
        trace_id = obj.trace_id
        session_no = int(obj.session_no)
        epoch_sec = int(obj.epoch_seconds or 30)
        edf_bytes = bytes(obj.inline_bytes) if obj.HasField("inline_bytes") else None
        if not edf_bytes:
            return

        chunks = splitter.split(edf_bytes)
        key = f"{session_no}:{trace_id}"
        for ch in chunks:
            samples_per_epoch = int(round(epoch_sec * float(ch.sampling_rate_hz)))
            n_epochs = ch.data.shape[1] // samples_per_epoch
            for e in range(n_epochs):
                start = e * samples_per_epoch
                end = start + samples_per_epoch
                segment = ch.data[:, start:end]
                recorded_at_ms = int((ch.start_at + timedelta(seconds=e * epoch_sec)).timestamp() * 1000)
                msg_out = pb.BrainwaveSplitEpoch(  # type: ignore[attr-defined]
                    trace_id=trace_id,
                    session_no=session_no,
                    epoch_index=int(e),
                    epoch_end_index=int(n_epochs - 1),
                    recorded_at_ms=recorded_at_ms,
                    sampling_rate_hz=float(ch.sampling_rate_hz),
                    num_channels=int(segment.shape[0]),
                    sample_count=int(segment.shape[1]),
                    payload=segment.astype(np.float32).tobytes(),
                )
                headers = {"trace_id": trace_id, "session_no": str(session_no), "version": "1", "content-type": "application/x-protobuf;msg=BrainwaveSplitEpoch"}
                producer.send_bytes(topic_out, key=key, value_bytes=msg_out.SerializeToString(), headers=headers)

        logger.info("splitter_sent", extra={"trace_id": trace_id, "session_no": session_no, "epochs": int(n_epochs)})

    runner = KafkaStageRunner(
        consumer=consumer,
        start_producer=producer.start,
        stop_producer=producer.stop,
        handle_message=_handle,
    )
    return await runner.run_forever()


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())



