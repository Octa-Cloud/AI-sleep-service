from __future__ import annotations

from app.common.kafka.interfaces import KafkaMessageHandler
import logging
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
        self._logger = logging.getLogger("report.dbwriter.daily")

    @session_scope
    def __call__(self, value: bytes, headers: dict[str, str], session=None) -> None:  # type: ignore[override]
        # Parse body directly without relying on Kafka headers
        obj = rp.DailyReportPersistRequest()  # type: ignore[attr-defined]
        obj.ParseFromString(value)

        from datetime import datetime, timezone
        created_at = datetime.fromtimestamp(obj.created_at_ms / 1000, tz=timezone.utc)
        trace_id = headers.get("trace_id", "")

        dr = session.get(DailyReport, obj.session_no)
        self._logger.info(
            f"daily_db_writer_recv session_no={int(obj.session_no)} user_no={int(obj.user_no)} trace_id={trace_id}"
        )
        if dr is None:
            self._logger.warning(
                "daily_missing_placeholder", extra={"session_no": int(obj.session_no), "user_no": int(obj.user_no), "trace_id": trace_id}
            )
            return
        # Log incoming aggregates to diagnose empties
        self._logger.info(
            (
                f"daily_db_writer_values session_no={int(obj.session_no)} user_no={int(obj.user_no)} "
                f"score={int(obj.score or 0)} total_min={int(obj.total_sleep_minutes or 0)} "
                f"deep_min={int(obj.deep_sleep_minutes or 0)} light_min={int(obj.light_sleep_minutes or 0)} rem_min={int(obj.rem_sleep_minutes or 0)} "
                f"deep_ratio={float(obj.deep_sleep_ratio or 0.0)} light_ratio={float(obj.light_sleep_ratio or 0.0)} rem_ratio={float(obj.rem_sleep_ratio or 0.0)} "
                f"n_details={len(getattr(obj, 'details', []))}"
            )
        )

        # Update final score via service upsert (memo must not be overwritten)
        self._svc.update_final(
            sleep_session_no=int(obj.session_no),
            user_no=int(obj.user_no),
            memo=None,
            score=int(obj.score or 0),
            session=session,
        )

        # SleepTimeDetail: create or update directly in DB writer
        std = session.get(SleepTimeDetail, int(obj.session_no))
        if std is None:
            std = SleepTimeDetail(
                sleep_session_no=int(obj.session_no),
                deep_sleep_minutes=0,
                light_sleep_minutes=0,
                rem_sleep_minutes=0,
                deep_sleep_ratio=0.0,
                light_sleep_ratio=0.0,
                rem_sleep_ratio=0.0,
            )
            session.add(std)
        std.deep_sleep_minutes = int(obj.deep_sleep_minutes or 0)
        std.light_sleep_minutes = int(obj.light_sleep_minutes or 0)
        std.rem_sleep_minutes = int(obj.rem_sleep_minutes or 0)
        std.deep_sleep_ratio = float(obj.deep_sleep_ratio or 0.0)
        std.light_sleep_ratio = float(obj.light_sleep_ratio or 0.0)
        std.rem_sleep_ratio = float(obj.rem_sleep_ratio or 0.0)

        session.query(AnalysisStep).filter(AnalysisStep.analysis_detail_no.in_(
            session.query(AnalysisDetail.analysis_detail_no).filter(AnalysisDetail.sleep_session_no == int(obj.session_no))
        )).delete(synchronize_session=False)
        session.query(AnalysisDetail).filter(AnalysisDetail.sleep_session_no == int(obj.session_no)).delete(synchronize_session=False)

        # AnalysisDetail/Step: replace directly in DB writer
        session.query(AnalysisStep).filter(AnalysisStep.analysis_detail_no.in_(
            session.query(AnalysisDetail.analysis_detail_no).filter(AnalysisDetail.sleep_session_no == int(obj.session_no))
        )).delete(synchronize_session=False)
        session.query(AnalysisDetail).filter(AnalysisDetail.sleep_session_no == int(obj.session_no)).delete(synchronize_session=False)

        n_details = 0
        n_steps_total = 0
        for d in obj.details:
            ad = AnalysisDetail(
                sleep_session_no=int(obj.session_no),
                title=str(d.title),
                description=str(d.description),
                difficulty=Difficulty[d.difficulty.name],
                effect=Effect[d.effect.name],
            )
            session.add(ad)
            session.flush()
            for s in d.steps:
                session.add(AnalysisStep(
                    analysis_detail_no=int(ad.analysis_detail_no),
                    step_index=int(s.step_index),
                    content=str(s.content),
                ))
                n_steps_total += 1
            n_details += 1
        self._logger.info(
            (
                f"daily_db_writer_analysis_applied session_no={int(obj.session_no)} "
                f"n_details={n_details} n_steps_total={n_steps_total}"
            )
        )
        self._logger.info(
            f"daily_db_writer_updated session_no={int(obj.session_no)} user_no={int(obj.user_no)} trace_id={trace_id}"
        )


