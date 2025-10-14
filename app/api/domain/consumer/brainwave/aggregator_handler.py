from __future__ import annotations

import logging
from typing import Any, Dict

from app.common.kafka.interfaces import KafkaMessageHandler
from app.common import config

try:
    from app.common.kafka.dto import brainwave_pb2 as pb  # type: ignore
except Exception:  # pragma: no cover
    pb = None  # type: ignore


class BrainwaveAggregatorHandler(KafkaMessageHandler):
    """Aggregates analyzed epochs per trace_id and emits persist requests via a provided producer."""

    def __init__(self, producer) -> None:
        self._producer = producer
        self._use_proto = config.KAFKA_PROTOBUF_ENABLED and pb is not None
        self._logger = logging.getLogger("brainwave.handler.aggregator")
        self._buf: Dict[str, Dict[str, Any]] = {}

    def __call__(self, value: bytes, headers: dict[str, str]) -> None:
        buf = None
        try:
            print(f"Aggregator received message: trace_id={headers.get('trace_id', 'unknown')}")
            
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
                import json
                message = json.loads((value or b"{}").decode("utf-8"))
                trace_id = str(message.get("trace_id"))
                session_no = int(message.get("session_no"))
                idx = int(message.get("epoch_index"))
                end_idx = int(message.get("epoch_end_index"))
                level = int(message.get("level", 0))
                recorded_at = str(message.get("recorded_at"))

            print(f"Aggregator processing: trace_id={trace_id}, epoch_idx={idx}, end_idx={end_idx}, level={level}")
            
            buf = self._buf.setdefault(trace_id, {"session_no": session_no, "end": end_idx, "items": {}})
            buf["items"][idx] = {"recorded_at": recorded_at, "level": level}
            
            print(f"Aggregator buffer: trace_id={trace_id}, collected={len(buf['items'])}, needed={buf['end'] + 1}")
        except Exception as e:
            print(f"Aggregator ERROR: {e}")
            import traceback
            traceback.print_exc()
            return  # 에러 발생 시 처리 중단

        if buf and len(buf["items"]) >= (buf["end"] + 1):
            # All epochs collected -> emit persist request
            out_topic = config.TOPIC_BRAINWAVE_PERSIST_REQUESTS
            levels = [
                {"epoch_index": i, "recorded_at": buf["items"][i]["recorded_at"], "level": buf["items"][i]["level"]}
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
                if hasattr(self._producer, "send_bytes"):
                    self._producer.send_bytes(out_topic, key=key, value_bytes=msg_out.SerializeToString(), headers=hdrs)  # type: ignore[misc]
            else:
                out_msg = {"trace_id": trace_id, "session_no": session_no, "levels": levels}
                hdrs["content-type"] = "application/json;msg=BrainwavePersistRequest"
                if hasattr(self._producer, "send"):
                    self._producer.send(out_topic, key=key, value=out_msg, headers=hdrs)  # type: ignore[misc]
            # Clear buffer
            self._buf.pop(trace_id, None)
            self._logger.info("aggregated_emit_persist", extra={"trace_id": trace_id, "session_no": session_no, "n_epochs": len(levels)})


