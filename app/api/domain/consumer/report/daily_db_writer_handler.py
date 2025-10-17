from __future__ import annotations

from app.common.kafka.interfaces import KafkaMessageHandler
from app.api.common.decorator.session_scope import session_scope
from app.api.domain.domain.entity.sleep_session_entity import DailyReport, SleepTimeDetail, AnalysisDetail, AnalysisStep, Difficulty, Effect
from app.api.domain.application.service.daily_report.daily_report_service import DailyReportService

try:
    from app.common.kafka.dto import report_pb2 as rp  # type: ignore
except Exception:  # pragma: no cover
    rp = None  # type: ignore


class DailyReportDbWriterHandler(KafkaMessageHandler):
    def __init__(self, service: DailyReportService) -> None:
        if rp is None:
            raise RuntimeError("Protobuf stubs not generated for report. Run scripts/gen_protos.py")
        self._svc = service

    @session_scope
    def __call__(self, value: bytes, headers: dict[str, str], session=None) -> None:  # type: ignore[override]
        content_type = headers.get("content-type", "")
        if "DailyReportPersistRequest" not in content_type:
            return
        obj = rp.DailyReportPersistRequest()  # type: ignore[attr-defined]
        obj.ParseFromString(value)

        from datetime import datetime, timezone
        created_at = datetime.fromtimestamp(obj.created_at_ms / 1000, tz=timezone.utc)
        trace_id = headers.get("trace_id", "")

        dr = session.get(DailyReport, obj.session_no)
        if dr is None:
            import logging
            logging.getLogger("report.dbwriter").warning(
                "daily_missing_placeholder", extra={"session_no": int(obj.session_no), "user_no": int(obj.user_no), "trace_id": trace_id}
            )
            return
        # Update memo/score via service upsert (allow_update=True)
        self._svc.update_final(
            sleep_session_no=int(obj.session_no),
            user_no=int(obj.user_no),
            memo=str(obj.memo or ""),
            score=int(obj.score or 0),
            session=session,
        )

        # Delegate sleep time detail update to service
        self._svc.update_sleep_time_detail(
            sleep_session_no=int(obj.session_no),
            deep_sleep_minutes=int(obj.deep_sleep_minutes or 0),
            light_sleep_minutes=int(obj.light_sleep_minutes or 0),
            rem_sleep_minutes=int(obj.rem_sleep_minutes or 0),
            deep_sleep_ratio=float(obj.deep_sleep_ratio or 0.0),
            light_sleep_ratio=float(obj.light_sleep_ratio or 0.0),
            rem_sleep_ratio=float(obj.rem_sleep_ratio or 0.0),
            session=session,
        )

        session.query(AnalysisStep).filter(AnalysisStep.analysis_detail_no.in_(
            session.query(AnalysisDetail.analysis_detail_no).filter(AnalysisDetail.sleep_session_no == int(obj.session_no))
        )).delete(synchronize_session=False)
        session.query(AnalysisDetail).filter(AnalysisDetail.sleep_session_no == int(obj.session_no)).delete(synchronize_session=False)

        # Delegate analysis replace to service
        details_payload = []
        for d in obj.details:
            steps_payload = [(int(s.step_index), str(s.content)) for s in d.steps]
            details_payload.append((str(d.title), str(d.description), Difficulty[d.difficulty.name], Effect[d.effect.name], steps_payload))
        self._svc.replace_analysis(
            sleep_session_no=int(obj.session_no),
            details=details_payload,
            session=session,
        )


