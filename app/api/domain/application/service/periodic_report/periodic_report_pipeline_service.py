from __future__ import annotations

from datetime import date, timedelta
from typing import List
import logging

from app.api.domain.application.service.daily_report.daily_report_service import DailyReportService
from app.api.domain.application.service.periodic_report.periodic_report_agent_service import PeriodicReportAgentService
from app.api.domain.infra.repository.sleep_session_repository_impl import SqlAlchemySleepSessionRepository
from app.api.common.decorator.session_scope import session_scope

try:
    from app.common.kafka.dto import report_pb2 as rp  # type: ignore
except Exception:  # pragma: no cover
    rp = None  # type: ignore


class PeriodicReportPipelineService:
    def __init__(self, daily_service: DailyReportService, agent_service: PeriodicReportAgentService) -> None:
        self._daily = daily_service
        self._agent = agent_service
        self._logger = logging.getLogger("report.pipeline.periodic")

    def _calc_period_start(self, base: date, duration_type: int) -> date:
        if duration_type == int(rp.DurationType.WEEKLY):  # type: ignore[attr-defined]
            days_since_sunday = (base.weekday() + 1) % 7
            return base - timedelta(days=days_since_sunday)
        return base.replace(day=1)

    @session_scope
    def _sum_bed_time_minutes(self, session_nos: List[int], session=None) -> int:
        repo = SqlAlchemySleepSessionRepository(session=session)
        total = 0
        for sn in session_nos:
            s = repo.find_by_id(int(sn))
            if s and s.finished_at and s.created_at:
                delta = (s.finished_at - s.created_at).total_seconds() / 60.0
                if delta > 0:
                    total += int(delta)
        return total

    async def build_persist_request(self, pri: "rp.PeriodicReportInput") -> "rp.PeriodicReportPersistRequest":  # type: ignore[name-defined]
        user_no = int(pri.user_no)
        base = date.fromisoformat(str(pri.sleep_date))
        duration_type = int(pri.duration_type)
        period_start = self._calc_period_start(base, duration_type)
        self._logger.info(
            (
                f"periodic_pipeline_build_start user_no={user_no} sleep_date={base.isoformat()} "
                f"duration_type={'WEEKLY' if duration_type == int(rp.DurationType.WEEKLY) else 'MONTHLY'} period_start={period_start.isoformat()}"
            )
        )

        reports = self._daily.get_range(user_no, period_start, base)
        sleep_session_count = len(reports)
        total_score = sum(int(r.score or 0) for r in reports)
        total_sleep_time_minutes = sum(int(r.total_sleep_minutes or 0) for r in reports)
        total_deep = sum(int(r.deep_sleep_minutes or 0) for r in reports)
        total_light = sum(int(r.light_sleep_minutes or 0) for r in reports)
        total_rem = sum(int(r.rem_sleep_minutes or 0) for r in reports)

        total_bed = self._sum_bed_time_minutes([int(r.sleep_session_no) for r in reports])

        payload = {
            "user_no": user_no,
            "period_start_date": period_start.isoformat(),
            "period_end_date": base.isoformat(),
            "total_score": total_score,
            "total_sleep_time_minutes": total_sleep_time_minutes,
            "total_deep_sleep_time_minutes": total_deep,
            "total_light_sleep_time_minutes": total_light,
            "total_rem_sleep_time_minutes": total_rem,
            "sleep_session_count": sleep_session_count,
        }
        try:
            daily_block, analysis_block = await self._agent.analyze(payload)
        except Exception:
            # include stack trace for precise diagnosis
            self._logger.exception(
                f"periodic_pipeline_agent_error user_no={user_no} period_start={period_start.isoformat()} payload_keys={list(payload.keys())}"
            )
            daily_block, analysis_block = {}, {}

        improvement = (analysis_block or {}).get("improvement", "") if analysis_block else ""
        weakness = (analysis_block or {}).get("weakness", "") if analysis_block else ""
        recommendation = (analysis_block or {}).get("recommendation", "") if analysis_block else ""
        score_prediction_description = (analysis_block or {}).get("score_prediction_description", "") if analysis_block else ""
        points = []
        for pt in (analysis_block or {}).get("points", []) or []:
            try:
                points.append(rp.ScorePredictionPoint(date_index=str(pt.get("date_index")), score=int(pt.get("score", 0))))  # type: ignore[attr-defined]
            except Exception:
                continue

        out = rp.PeriodicReportPersistRequest(  # type: ignore[attr-defined]
            user_no=user_no,
            duration_type=pri.duration_type,
            period_started_at=period_start.isoformat(),
            sleep_session_count=sleep_session_count,
            total_score=total_score,
            total_sleep_time_minutes=total_sleep_time_minutes,
            total_bed_time_minutes=int(total_bed),
            total_deep_sleep_time_minutes=total_deep,
            total_light_sleep_time_minutes=total_light,
            total_rem_sleep_time_minutes=total_rem,
            improvement=str(improvement or ""),
            weakness=str(weakness or ""),
            recommendation=str(recommendation or ""),
            score_prediction_description=str(score_prediction_description or ""),
            points=points,
        )
        self._logger.info(
            (
                f"periodic_pipeline_build_done user_no={user_no} period_start={period_start.isoformat()} "
                f"sleep_session_count={sleep_session_count} total_score={total_score}"
            )
        )
        return out


