from __future__ import annotations

from datetime import date
from typing import List

from app.api.common.decorator.session_scope import session_scope
from app.api.domain.domain.entity.periodic_report_entity import DurationType as PRDurationType, PeriodicReport, ScorePredictionPoint
from app.api.domain.infra.repository.periodic_report_repository_impl import SqlAlchemyPeriodicReportRepository


class PeriodicReportService:
    def __init__(self, repo_factory) -> None:
        self._repo_factory = repo_factory

    @session_scope
    def upsert_report(
        self,
        user_no: int,
        duration_type: PRDurationType,
        period_started_at: date,
        sleep_session_count: int,
        total_score: int,
        total_sleep_time_minutes: int,
        total_bed_time_minutes: int,
        total_deep_sleep_time_minutes: int,
        total_light_sleep_time_minutes: int,
        total_rem_sleep_time_minutes: int,
        improvement: str | None,
        weakness: str | None,
        recommendation: str | None,
        score_prediction_description: str | None,
        points: List[tuple[date, int]] | None,
        session=None,
    ) -> None:
        repo = self._repo_factory(session=session)
        entity = PeriodicReport(
            user_no=int(user_no),
            duration_type=duration_type,
            period_started_at=period_started_at,
            sleep_session_count=int(sleep_session_count),
            total_score=int(total_score),
            total_sleep_time=int(total_sleep_time_minutes),
            total_bed_time_minutes=int(total_bed_time_minutes),
            total_deep_sleep_time_minutes=int(total_deep_sleep_time_minutes),
            total_light_sleep_time_minutes=int(total_light_sleep_time_minutes),
            total_rem_sleep_time_minutes=int(total_rem_sleep_time_minutes),
            improvement=str(improvement or ""),
            weakness=str(weakness or ""),
            recommendation=str(recommendation or ""),
            score_prediction_description=str(score_prediction_description or ""),
        )
        repo.upsert(entity, points or [])


