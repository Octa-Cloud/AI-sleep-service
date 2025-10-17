from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from aiokafka import AIOKafkaConsumer

from app.common import config
from app.common.kafka.producer import KafkaProducerClient
from app.api.domain.worker.common.runner import KafkaStageRunner
from app.api.common.tsid import generate_int as generate_tsid_int
from app.api.domain.application.service.daily_report.daily_report_agent_service import DailyReportAgentService
from app.api.domain.application.service.daily_report.daily_report_pipeline_service import DailyReportPipelineService

try:
    from app.common.kafka.dto import report_pb2 as rp  # type: ignore
except Exception:  # pragma: no cover
    rp = None  # type: ignore


async def run() -> int:
    if rp is None:
        raise RuntimeError("Protobuf stubs not generated for report. Run scripts/gen_protos.py before starting the worker.")

    brokers = config.KAFKA_BROKERS
    topic_in = config.TOPIC_DAILY_REPORT_INPUT
    topic_out = config.TOPIC_DAILY_REPORT_PERSIST_REQUESTS
    group_id = config.GROUP_DAILY_REPORT_WORKER

    # SASL/SSL configuration for Confluent Cloud
    import ssl
    consumer_config = {
        "bootstrap_servers": brokers,
        "group_id": group_id,
        "enable_auto_commit": False,
        "auto_offset_reset": "earliest",
    }

    security_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL")
    if security_protocol:
        consumer_config["security_protocol"] = security_protocol
        consumer_config["sasl_mechanism"] = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
        consumer_config["sasl_plain_username"] = os.getenv("KAFKA_SASL_USERNAME", "")
        consumer_config["sasl_plain_password"] = os.getenv("KAFKA_SASL_PASSWORD", "")
        if "SSL" in security_protocol:
            ssl_context = ssl.create_default_context()
            consumer_config["ssl_context"] = ssl_context

    consumer = AIOKafkaConsumer(
        topic_in,
        **consumer_config
    )
    producer = KafkaProducerClient(brokers)

    pipeline = DailyReportPipelineService(DailyReportAgentService())

    async def _handle(value: bytes) -> None:
        obj = rp.DailyReportInput()  # type: ignore[attr-defined]
        obj.ParseFromString(value)

        session_no = int(obj.session_no)
        user_no = int(obj.user_no)
        created_at_ms = int(obj.created_at_ms)
        sleep_date_str = str(obj.sleep_date)

        persist = await pipeline.build_persist_request(obj)

        trace_id = str(generate_tsid_int())
        key = f"{session_no}:{trace_id}"
        headers = {
            "trace_id": trace_id,
            "session_no": str(session_no),
            "user_no": str(user_no),
            "sleep_date": sleep_date_str,
            "version": "1",
            "content-type": "application/x-protobuf;msg=DailyReportPersistRequest",
        }
        producer.send_bytes(topic_out, key=key, value_bytes=persist.SerializeToString(), headers=headers)

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


