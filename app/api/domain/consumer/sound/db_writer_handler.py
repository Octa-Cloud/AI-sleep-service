from __future__ import annotations

import logging
from typing import List
from datetime import datetime, timezone

from app.common.kafka.interfaces import KafkaMessageHandler
from app.api.domain.domain.entity.analyzed_data_entity import SoundEvent, SoundEventType
from app.api.common.tsid import generate_int as generate_tsid_int


class SoundDbWriterHandler(KafkaMessageHandler):
    def __init__(self, repo_service) -> None:
        self._service = repo_service

    def __call__(self, value: bytes, headers: dict[str, str]) -> None:
        from app.common.kafka.dto import sound_pb2 as pb  # type: ignore
        logger = logging.getLogger("sound.dbwriter")
        entities: List[SoundEvent] = []

        # Try batch first
        try:
            req = pb.SoundPersistRequest()  # type: ignore[attr-defined]
            req.ParseFromString(value)
            events = list(req.events)
            if events:
                logger.info(
                    f"dbwriter_recv_batch trace_id={getattr(req, 'trace_id', '-')} count={len(events)}",
                    extra={"trace_id": getattr(req, "trace_id", "-"), "count": len(events)},
                )
                for ev in events:
                    recorded_at = datetime.fromtimestamp(ev.at_ms / 1000.0, tz=timezone.utc)
                    entities.append(
                        SoundEvent(
                            analyzed_sound_event_no=int(generate_tsid_int()),
                            sleep_session_no=ev.sleep_session_no or None,
                            event=SoundEventType(ev.event) if ev.event in SoundEventType.__members__ else None,
                            recorded_at=recorded_at,
                        )
                    )
                try:
                    self._service.save_events(entities)
                except Exception:
                    logger.exception("dbwriter_save_error", extra={"trace_id": getattr(req, 'trace_id', '-'), "count": len(entities)})
                    return
                return
        except Exception:
            logger.exception("dbwriter_batch_error")

        # Single event fallback
        try:
            event = pb.SoundAnalyzedEvent()
            event.ParseFromString(value)
        except Exception:
            logger.exception("dbwriter_single_parse_error")
            return
        recorded_at = datetime.fromtimestamp(event.at_ms / 1000.0, tz=timezone.utc)
        entities.append(
            SoundEvent(
                analyzed_sound_event_no=int(generate_tsid_int()),
                sleep_session_no=event.sleep_session_no or None,
                event=SoundEventType(event.event) if event.event in SoundEventType.__members__ else None,
                recorded_at=recorded_at,
            )
        )
        try:
            self._service.save_events(entities)
        except Exception:
            logger.exception("dbwriter_save_error", extra={"trace_id": event.trace_id, "count": len(entities)})
            return
        logger.info(f"dbwriter_recv_single trace_id={event.trace_id}")


