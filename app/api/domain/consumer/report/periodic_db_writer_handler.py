from __future__ import annotations

from app.common.kafka.interfaces import KafkaMessageHandler
import logging
from app.api.common.decorator.session_scope import session_scope
from app.api.domain.domain.entity.periodic_report_entity import DurationType as PRDurationType
from app.api.domain.application.service.periodic_report.periodic_report_service import PeriodicReportService

try:
    from app.common.kafka.dto import report_pb2 as rp  # type: ignore
except Exception:  # pragma: no cover
    rp = None  # type: ignore


class PeriodicReportDbWriterHandler(KafkaMessageHandler):
    def __init__(self, service: PeriodicReportService) -> None:
        if rp is None:
            raise RuntimeError("Protobuf stubs not generated for report. Run scripts/gen_protos.py")
        self._svc = service
        self._logger = logging.getLogger("report.dbwriter.periodic")

    @session_scope
    def __call__(self, value: bytes, headers: dict[str, str], session=None) -> None:  # type: ignore[override]
        # Parse body directly without relying on Kafka headers
        obj = rp.PeriodicReportPersistRequest()  # type: ignore[attr-defined]
        obj.ParseFromString(value)

        from datetime import date as _date
        period_start = _date.fromisoformat(str(obj.period_started_at))
        duration = PRDurationType[obj.duration_type.name]

        # Delegate to service
        self._logger.info(
            f"periodic_db_writer_recv user_no={int(obj.user_no)} "
            f"duration_type={obj.duration_type.name} period_started_at={str(obj.period_started_at)}"
        )
        points = []
        from datetime import date as _d
        for p in obj.points:
            try:
                points.append((_d.fromisoformat(str(p.date_index)), int(p.score or 0)))
            except Exception:
                continue

        self._svc.upsert_report(
            user_no=int(obj.user_no),
            duration_type=duration,
            period_started_at=period_start,
            sleep_session_count=int(obj.sleep_session_count or 0),
            total_score=int(obj.total_score or 0),
            total_sleep_time_minutes=int(obj.total_sleep_time_minutes or 0),
            total_bed_time_minutes=int(obj.total_bed_time_minutes or 0),
            total_deep_sleep_time_minutes=int(obj.total_deep_sleep_time_minutes or 0),
            total_light_sleep_time_minutes=int(obj.total_light_sleep_time_minutes or 0),
            total_rem_sleep_time_minutes=int(obj.total_rem_sleep_time_minutes or 0),
            improvement=str(obj.improvement or ""),
            weakness=str(obj.weakness or ""),
            recommendation=str(obj.recommendation or ""),
            score_prediction_description=str(obj.score_prediction_description or ""),
            points=points,
            session=session,
        )
        self._logger.info(
            f"periodic_db_writer_upserted user_no={int(obj.user_no)} "
            f"duration_type={obj.duration_type.name} period_started_at={str(obj.period_started_at)}"
        )
        return


