from __future__ import annotations

import os
import os
import asyncio
import logging
from typing import Any, Dict, List

from app.common.kafka.consumer import AsyncKafkaConsumerRunner
from app.common.kafka.producer import KafkaProducerClient
from app.common import config
from app.api.domain.application.service.brainwave.sleep_level_service import SleepLevelService
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData


try:
    from app.common.kafka.dto import brainwave_pb2 as pb  # type: ignore
except Exception:  # pragma: no cover
    pb = None  # type: ignore


class Consumers:
    def __init__(self, container=None) -> None:
        self._instances: list[AsyncKafkaConsumerRunner] = []
        self._container = container
        self._agg_buffer: Dict[str, Dict[str, Any]] = {}
        # Reuse app-wide producer started in app startup
        self._producer = getattr(container, "kafka_producer", None) if container is not None else None

    def start_all(self) -> None:
        brokers = config.KAFKA_BROKERS
        self._use_proto = config.KAFKA_PROTOBUF_ENABLED and pb is not None
        self._logger = logging.getLogger("brainwave.Consumers")
        self._logger.setLevel(logging.INFO)

        # Aggregator: consumes analyzed epoch and aggregates per trace_id
        analyzed_topic = config.TOPIC_BRAINWAVE_ANALYZED_EPOCH
        agg_group = config.GROUP_BRAINWAVE_AGGREGATOR
        dlq_topic = config.TOPIC_DLQ
        self._agg_started = asyncio.Event()
        self._instances.append(AsyncKafkaConsumerRunner(brokers, analyzed_topic, agg_group, handler=self._on_analyzed_epoch, dlq_topic=dlq_topic, started_event=self._agg_started))

        # DB-writer: consumes persist requests and writes to DB
        persist_topic = config.TOPIC_BRAINWAVE_PERSIST_REQUESTS
        db_group = config.GROUP_BRAINWAVE_DB_WRITER
        self._db_started = asyncio.Event()
        self._instances.append(AsyncKafkaConsumerRunner(brokers, persist_topic, db_group, handler=self._on_persist_request, dlq_topic=dlq_topic, started_event=self._db_started))

        loop = asyncio.get_event_loop()
        for c in self._instances:
            loop.create_task(c.start())

    async def wait_ready(self) -> None:
        if hasattr(self, "_agg_started") and hasattr(self, "_db_started"):
            await self._agg_started.wait()
            await self._db_started.wait()

    def stop_all(self) -> None:
        loop = asyncio.get_event_loop()
        for c in self._instances:
            loop.create_task(c.stop())
        self._instances.clear()

    # -------- Handlers ---------
    def _on_analyzed_epoch(self, value: bytes, headers: dict[str, str]) -> None:
        # Parse analyzed epoch (protobuf or JSON)
        if self._use_proto:
            obj = pb.BrainwaveAnalyzedEpoch()  # type: ignore[attr-defined]
            obj.ParseFromString(value)
            trace_id = obj.trace_id
            session_no = int(obj.session_no)
            idx = int(obj.epoch_index)
            end_idx = int(obj.epoch_end_index)
            level = int(obj.level)
            from datetime import datetime, timezone
            recorded_at = datetime.fromtimestamp(obj.recorded_at_ms / 1000, tz=timezone.utc).isoformat()
        else:
            message = self._parse_json(value)
            trace_id = str(message.get("trace_id"))
            session_no = int(message.get("session_no"))
            idx = int(message.get("epoch_index"))
            end_idx = int(message.get("epoch_end_index"))
            level = int(message.get("level"))
            recorded_at = str(message.get("recorded_at"))

        buf = self._agg_buffer.setdefault(trace_id, {"session_no": session_no, "end": end_idx, "items": {}})
        buf["items"][idx] = {"level": level, "recorded_at": recorded_at}

        if len(buf["items"]) >= (buf["end"] + 1):
            # All epochs collected -> emit persist request
            out_topic = os.getenv("TOPIC_BRAINWAVE_PERSIST_REQUESTS", "brainwave.persist.requests")
            levels = [
                {"epoch_index": i, "level": buf["items"][i]["level"], "recorded_at": buf["items"][i]["recorded_at"]}
                for i in sorted(buf["items"].keys())
            ]
            key = f"{session_no}:{trace_id}"
            hdrs = {"trace_id": trace_id, "session_no": str(session_no), "version": "1"}
            if self._use_proto:
                from datetime import datetime
                items = []
                for it in levels:
                    dt = datetime.fromisoformat(it["recorded_at"])  # naive or aware ok
                    items.append(pb.BrainwavePersistRequest.LevelItem(  # type: ignore[attr-defined]
                        epoch_index=int(it["epoch_index"]),
                        level=int(it["level"]),
                        recorded_at_ms=int(dt.timestamp() * 1000),
                    ))
                msg_out = pb.BrainwavePersistRequest(  # type: ignore[attr-defined]
                    trace_id=trace_id, session_no=session_no, levels=items
                )
                hdrs["content-type"] = "application/x-protobuf;msg=BrainwavePersistRequest"
                if self._producer is not None and hasattr(self._producer, "send_bytes"):
                    self._producer.send_bytes(out_topic, key=key, value_bytes=msg_out.SerializeToString(), headers=hdrs)  # type: ignore[misc]
            else:
                out_msg = {"trace_id": trace_id, "session_no": session_no, "n_epochs": len(levels), "levels": levels}
                hdrs["content-type"] = "application/json;msg=BrainwavePersistRequest"
                if self._producer is not None and hasattr(self._producer, "send"):
                    self._producer.send(out_topic, key=key, value=out_msg, headers=hdrs)  # type: ignore[misc]
            # Clear buffer
            self._agg_buffer.pop(trace_id, None)
            self._logger.info("aggregated_emit_persist", extra={"trace_id": trace_id, "session_no": session_no, "n_epochs": len(levels)})

    def _on_persist_request(self, value: bytes, headers: dict[str, str]) -> None:
        if self._container is None:
            return
        # parse persist request
        vo_list: List[SleepLevelData] = []
        if self._use_proto:
            obj = pb.BrainwavePersistRequest()  # type: ignore[attr-defined]
            obj.ParseFromString(value)
            session_no = int(obj.session_no)
            from datetime import datetime, timezone
            for it in obj.levels:
                vo_list.append(SleepLevelData(level=int(it.level), recorded_at=datetime.fromtimestamp(it.recorded_at_ms / 1000, tz=timezone.utc)))
        else:
            message = self._parse_json(value)
            session_no = int(message.get("session_no"))
            levels = message.get("levels") or []
            for item in levels:
                vo_list.append(SleepLevelData(level=int(item["level"]), recorded_at=self._parse_iso(item["recorded_at"])))

        # Use SleepLevelService via container
        svc: SleepLevelService = self._container.brainwave_sleeplevel  # type: ignore[attr-defined]
        entities = svc.data_to_entities(session_no, vo_list)
        svc.insert_bulk(entities)
        self._logger.info("dbwriter_saved", extra={"trace_id": headers.get("trace_id"), "session_no": session_no, "n_entities": len(entities)})

    @staticmethod
    def _parse_iso(value: str):
        from datetime import datetime
        return datetime.fromisoformat(value)

    @staticmethod
    def _parse_json(value: bytes) -> dict:
        import json
        return json.loads((value or b"{}").decode("utf-8"))


