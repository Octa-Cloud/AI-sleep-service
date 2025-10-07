from __future__ import annotations

from typing import List

from app.common.kafka.interfaces import KafkaMessageHandler
from app.api.domain.application.service.brainwave.sleep_level_service import SleepLevelService
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData

try:
    from app.common.kafka.dto import brainwave_pb2 as pb  # type: ignore
except Exception:  # pragma: no cover
    pb = None  # type: ignore


class BrainwaveDbWriterHandler(KafkaMessageHandler):
    def __init__(self, service: SleepLevelService, use_protobuf: bool) -> None:
        self._service = service
        self._use_proto = use_protobuf and pb is not None

    def __call__(self, value: bytes, headers: dict[str, str]) -> None:
        vo_list: List[SleepLevelData] = []
        if self._use_proto:
            obj = pb.BrainwavePersistRequest()  # type: ignore[attr-defined]
            obj.ParseFromString(value)
            session_no = int(obj.session_no)
            from datetime import datetime, timezone
            for it in obj.levels:
                vo_list.append(SleepLevelData(level=int(it.level), recorded_at=datetime.fromtimestamp(it.recorded_at_ms / 1000, tz=timezone.utc)))
        else:
            import json
            message = json.loads((value or b"{}").decode("utf-8"))
            session_no = int(message.get("session_no"))
            levels = message.get("levels") or []
            from datetime import datetime
            for item in levels:
                vo_list.append(SleepLevelData(level=int(item["level"]), recorded_at=datetime.fromisoformat(item["recorded_at"])))

        entities = self._service.data_to_entities(session_no, vo_list)
        self._service.insert_bulk(entities)


