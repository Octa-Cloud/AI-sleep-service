from __future__ import annotations

import asyncio
import os
import logging
from datetime import timedelta

from aiokafka import AIOKafkaConsumer

from app.common import config
from app.common.kafka.producer import KafkaProducerClient
from app.api.domain.worker.common.runner import KafkaStageRunner
from app.api.common.tsid import generate_int as generate_tsid_int
from app.api.domain.application.service.periodic_report.periodic_report_agent_service import PeriodicReportAgentService
from app.api.domain.application.service.daily_report.daily_report_service import DailyReportService
from app.api.domain.infra.repository.daily_report_repository_impl import SqlAlchemyDailyReportRepository
from app.api.domain.application.service.periodic_report.periodic_report_pipeline_service import PeriodicReportPipelineService

try:
    from app.common.kafka.dto import report_pb2 as rp  # type: ignore
except Exception:  # pragma: no cover
    rp = None  # type: ignore


async def run() -> int:
    if rp is None:
        raise RuntimeError("Protobuf stubs not generated for report. Run scripts/gen_protos.py before starting the worker.")

    brokers = config.KAFKA_BROKERS
    topic_in = config.TOPIC_PERIODIC_REPORT_INPUT
    topic_out = config.TOPIC_PERIODIC_REPORT_PERSIST_REQUESTS
    group_id = config.GROUP_PERIODIC_REPORT_WORKER

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

    def _repo_factory(session=None):
        return SqlAlchemyDailyReportRepository(session=session)
    pipeline = PeriodicReportPipelineService(DailyReportService(repo_factory=_repo_factory), PeriodicReportAgentService())
    logger = logging.getLogger("report.worker.periodic")

    async def _handle(value: bytes) -> None:
        try:
            obj = rp.PeriodicReportInput()  # type: ignore[attr-defined]
            obj.ParseFromString(value)

            persist = await pipeline.build_persist_request(obj)

            trace_id = str(generate_tsid_int())
            key = f"{obj.session_no}:{trace_id}"
            user_no = int(obj.user_no)
            sleep_date = str(obj.sleep_date)
            headers = {
                "trace_id": trace_id,
                "session_no": str(int(obj.session_no)),
                "user_no": str(user_no),
                "sleep_date": sleep_date,
                "version": "1",
                "content-type": "application/x-protobuf;msg=PeriodicReportPersistRequest",
            }
            # Map enum to readable name without relying on .name
            dtype_name = "WEEKLY" if int(obj.duration_type) == int(rp.DurationType.WEEKLY) else "MONTHLY"
            logger.info(
                "periodic_worker_recv",
                extra={
                    "session_no": int(obj.session_no),
                    "user_no": user_no,
                    "sleep_date": sleep_date,
                    "duration_type": dtype_name,
                    "topic": topic_in,
                },
            )
            producer.send_bytes(topic_out, key=key, value_bytes=persist.SerializeToString(), headers=headers)
            logger.info(
                "periodic_worker_emit_persist",
                extra={
                    "session_no": int(obj.session_no),
                    "user_no": user_no,
                    "sleep_date": sleep_date,
                    "trace_id": trace_id,
                    "topic": topic_out,
                },
            )
        except Exception:
            # include full traceback for diagnosis
            logger.exception("periodic_worker_error", extra={"topic": topic_in})

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


