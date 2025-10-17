from __future__ import annotations

from datetime import date
from typing import List, Tuple

from sqlalchemy.orm import Session

from app.api.domain.domain.repository.periodic_report_repository import PeriodicReportRepository
from app.api.domain.domain.entity.periodic_report_entity import PeriodicReport, ScorePredictionPoint


class SqlAlchemyPeriodicReportRepository(PeriodicReportRepository):
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, entity: PeriodicReport, points: List[Tuple[date, int]]) -> None:
        # Find existing by unique key (user_no, duration_type, period_started_at)
        existing = (
            self._session.query(PeriodicReport)
            .filter(
                PeriodicReport.user_no == int(entity.user_no),
                PeriodicReport.duration_type == entity.duration_type,
                PeriodicReport.period_started_at == entity.period_started_at,
            )
            .one_or_none()
        )
        if existing is None:
            self._session.add(entity)
            self._session.flush()
            pr = entity
        else:
            # update fields
            existing.sleep_session_count = int(entity.sleep_session_count)
            existing.total_score = int(entity.total_score)
            existing.total_sleep_time = int(entity.total_sleep_time)
            existing.total_bed_time_minutes = int(entity.total_bed_time_minutes)
            existing.total_deep_sleep_time_minutes = int(entity.total_deep_sleep_time_minutes)
            existing.total_light_sleep_time_minutes = int(entity.total_light_sleep_time_minutes)
            existing.total_rem_sleep_time_minutes = int(entity.total_rem_sleep_time_minutes)
            existing.improvement = str(entity.improvement or "")
            existing.weakness = str(entity.weakness or "")
            existing.recommendation = str(entity.recommendation or "")
            existing.score_prediction_description = str(entity.score_prediction_description or "")
            self._session.flush()
            pr = existing

        # Replace points
        self._session.query(ScorePredictionPoint).filter(ScorePredictionPoint.periodic_report_no == pr.periodic_report_no).delete(synchronize_session=False)
        for dt, sc in points or []:
            self._session.add(ScorePredictionPoint(periodic_report_no=int(pr.periodic_report_no), date_index=dt, score=int(sc)))


