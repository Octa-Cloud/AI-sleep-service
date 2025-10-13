from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

import numpy as np
import logging
from aiokafka import AIOKafkaConsumer
from app.common import config

try:
    from app.common.kafka.dto import brainwave_pb2 as pb  # type: ignore
except Exception:  # pragma: no cover
    pb = None  # type: ignore

from app.common.kafka.producer import KafkaProducerClient
from app.api.domain.worker.common.runner import KafkaStageRunner
from app.api.domain.application.service.brainwave.brainwave_analyzer_service import BrainwaveAnalyzerService
from app.api.domain.domain.vo.chunked_data_value_object import BrainwaveChunkData


async def run() -> int:
    brokers = config.KAFKA_BROKERS
    topic_in = config.TOPIC_BRAINWAVE_SPLIT_EPOCHS
    topic_out = config.TOPIC_BRAINWAVE_ANALYZED_EPOCH
    group_id = config.GROUP_BRAINWAVE_ANALYZER
    use_proto = True
    if pb is None:
        raise RuntimeError("Protobuf stubs not generated. Run scripts/gen_protos.sh before starting the worker.")

    # SASL/SSL configuration for Confluent Cloud
    import ssl
    consumer_config = {
        "bootstrap_servers": brokers,
        "group_id": group_id,
        "enable_auto_commit": False,
        "auto_offset_reset": "earliest",
    }
    
    # Add SASL authentication if configured
    security_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL")
    if security_protocol:
        consumer_config["security_protocol"] = security_protocol
        consumer_config["sasl_mechanism"] = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
        consumer_config["sasl_plain_username"] = os.getenv("KAFKA_SASL_USERNAME", "")
        consumer_config["sasl_plain_password"] = os.getenv("KAFKA_SASL_PASSWORD", "")
        
        # SSL context for SASL_SSL
        if "SSL" in security_protocol:
            ssl_context = ssl.create_default_context()
            consumer_config["ssl_context"] = ssl_context
    
    consumer = AIOKafkaConsumer(
        topic_in,
        **consumer_config
    )
    producer = KafkaProducerClient(brokers)
    analyzer = BrainwaveAnalyzerService()

    await producer.start()
    logger = logging.getLogger("brainwave.Analyzer")
    logger.setLevel(logging.INFO)
    async def _handle(value: bytes) -> None:
        trace_id: str
        session_no: int
        epoch_index: int
        epoch_end_index: int
        recorded_at: datetime
        sampling_rate_hz: float
        arr: np.ndarray

        obj = pb.BrainwaveSplitEpoch()  # type: ignore[attr-defined]
        obj.ParseFromString(value)
        trace_id = obj.trace_id
        session_no = int(obj.session_no)
        epoch_index = int(obj.epoch_index)
        epoch_end_index = int(obj.epoch_end_index)
        recorded_at = datetime.fromtimestamp(obj.recorded_at_ms / 1000, tz=timezone.utc)
        sampling_rate_hz = float(obj.sampling_rate_hz)
        num_channels = int(obj.num_channels or 2)
        sample_count = int(obj.sample_count)
        arr = np.frombuffer(obj.payload, dtype=np.float32).reshape(num_channels, sample_count)

        chunk = BrainwaveChunkData(data=arr, start_at=recorded_at, sampling_rate_hz=sampling_rate_hz)
        vo_list = analyzer.analyze([chunk])
        predicted_level = int(vo_list[0].level) if vo_list else 0

        msg_out = pb.BrainwaveAnalyzedEpoch(  # type: ignore[attr-defined]
            trace_id=trace_id,
            session_no=session_no,
            epoch_index=epoch_index,
            epoch_end_index=epoch_end_index,
            level=predicted_level,
            recorded_at_ms=int(recorded_at.timestamp() * 1000),
        )
        key = f"{session_no}:{trace_id}"
        headers = {"trace_id": trace_id, "session_no": str(session_no), "version": "1", "content-type": "application/x-protobuf;msg=BrainwaveAnalyzedEpoch"}
        producer.send_bytes(topic_out, key=key, value_bytes=msg_out.SerializeToString(), headers=headers)
        logger.info("analyzer_sent", extra={"trace_id": trace_id, "session_no": session_no, "epoch_index": epoch_index, "level": predicted_level})

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


