from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.common.kafka.interfaces import KafkaMessageHandler
from app.common import config

try:
    from app.common.kafka.dto import sound_pb2 as pb  # type: ignore
except Exception:  # pragma: no cover
    pb = None  # type: ignore


class SoundAggregatorHandler(KafkaMessageHandler):
    """Aggregates analyzed sound events per trace_id and emits batch persist requests."""

    def __init__(self, producer) -> None:
        self._producer = producer
        self._use_proto = config.KAFKA_PROTOBUF_ENABLED and pb is not None
        self._logger = logging.getLogger("sound.handler.aggregator")
        self._buf: Dict[str, Dict[str, Any]] = {}

    def __call__(self, value: bytes, headers: dict[str, str]) -> None:
        if self._use_proto:
            obj = pb.SoundAnalyzedEvent()  # type: ignore[attr-defined]
            obj.ParseFromString(value)
            trace_id = obj.trace_id
            session_no = int(getattr(obj, "sleep_session_no", 0) or getattr(obj, "session_no", 0))
            idx = int(headers.get("epoch_index", "0"))
            end_idx = int(headers.get("epoch_end_index", "0"))
            at_ms = int(obj.at_ms)
            event = str(obj.event)
        else:
            import json
            message = json.loads((value or b"{}").decode("utf-8"))
            trace_id = str(message.get("trace_id"))
            session_no = int(message.get("sleep_session_no") or message.get("session_no", 0))
            idx = int(headers.get("epoch_index", "0"))
            end_idx = int(headers.get("epoch_end_index", "0"))
            at_ms = int(message.get("at_ms", 0))
            event = str(message.get("event", ""))
        self._logger.debug(f"agg_recv event={event}")
        buf = self._buf.setdefault(trace_id, {"session_no": session_no, "end": end_idx, "items": {}})
        buf["items"][idx] = {"at_ms": at_ms, "event": event}

        # Flush when all indices 0..end are present
        if len(buf["items"]) >= (buf["end"] + 1):
            out_topic = config.TOPIC_SOUND_PERSIST_REQUESTS
            items = [buf["items"][i] for i in sorted(buf["items"].keys())]
            key = f"{session_no}:{trace_id}"
            hdrs = {"trace_id": trace_id, "session_no": str(session_no), "version": "1"}
            if self._use_proto:
                events: List[pb.SoundAnalyzedEvent] = []  # type: ignore[attr-defined]
                for i, it in enumerate(items):
                    events.append(
                        pb.SoundAnalyzedEvent(  # type: ignore[attr-defined]
                            sleep_session_no=session_no,
                            at_ms=int(it["at_ms"]),
                            event=str(it["event"]),
                            trace_id=trace_id,
                        )
                    )
                msg_out = pb.SoundPersistRequest(events=events, trace_id=trace_id)  # type: ignore[attr-defined]
                hdrs["content-type"] = "application/x-protobuf;msg=SoundPersistRequest"
                if hasattr(self._producer, "send_bytes"):
                    self._producer.send_bytes(out_topic, key=key, value_bytes=msg_out.SerializeToString(), headers=hdrs)  # type: ignore[misc]
            else:
                out_msg = {"trace_id": trace_id, "session_no": session_no, "events": [
                    {"at_ms": it["at_ms"], "event": it["event"]} for it in items
                ]}
                hdrs["content-type"] = "application/json;msg=SoundPersistRequest"
                if hasattr(self._producer, "send"):
                    self._producer.send(out_topic, key=key, value=out_msg, headers=hdrs)  # type: ignore[misc]
            # Clear buffer
            self._buf.pop(trace_id, None)
            self._logger.debug(f"agg_emit_persist n_events={len(items)}")


