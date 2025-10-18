from __future__ import annotations

from datetime import date
from typing import List, Optional
import logging
from app.api.common.decorator.session_scope import session_scope
from app.api.domain.domain.entity.sleep_session_entity import DailyReport as DailyReportEntity

from app.api.domain.domain.repository.daily_report_repository import DailyReportRepository
from app.api.domain.domain.vo.report_value_object import DailyReportData
from app.api.domain.domain.entity.sleep_session_entity import SleepTimeDetail, AnalysisDetail, AnalysisStep, Difficulty, Effect


class DailyReportService:
    def __init__(self, repo_factory) -> None:
        self._repo_factory = repo_factory
        self._logger = logging.getLogger("daily.service")

    @session_scope
    def get_by_date(self, user_no: int, sleep_date: date, session=None) -> Optional[DailyReportData]:
        repo: DailyReportRepository = self._repo_factory(session=session) if session is not None else self._repo_factory()
        try:
            self._logger.info("daily_get_by_date", extra={"user_no": int(user_no), "sleep_date": sleep_date.isoformat()})
        except Exception:
            pass
        return repo.get_by_date(int(user_no), sleep_date)

    @session_scope
    def get_range(self, user_no: int, start_date: date, end_date: date, session=None) -> List[DailyReportData]:
        repo: DailyReportRepository = self._repo_factory(session=session) if session is not None else self._repo_factory()
        try:
            self._logger.info("daily_get_range", extra={"user_no": int(user_no), "start_date": start_date.isoformat(), "end_date": end_date.isoformat()})
        except Exception:
            pass
        return repo.get_range(int(user_no), start_date, end_date)

    @session_scope
    def insert_placeholder(self, sleep_session_no: int, user_no: int, created_at, session=None) -> None:
        repo: DailyReportRepository = self._repo_factory(session=session) if session is not None else self._repo_factory()
        repo.upsert_daily_report(int(sleep_session_no), int(user_no), created_at, memo=None, score=None)
        try:
            self._logger.info("daily_placeholder_inserted", extra={"session_no": int(sleep_session_no), "user_no": int(user_no)})
        except Exception:
            pass

    @session_scope
    def update_final(self, sleep_session_no: int, user_no: int, memo: str | None, score: int | None, session=None) -> None:
        repo: DailyReportRepository = self._repo_factory(session=session) if session is not None else self._repo_factory()
        repo.upsert_daily_report(int(sleep_session_no), int(user_no), created_at=None, memo=memo, score=score)
        try:
            self._logger.info("daily_final_updated", extra={"session_no": int(sleep_session_no), "user_no": int(user_no), "score": int(score or 0)})
        except Exception:
            pass

    @session_scope
    def update_sleep_time_detail(
        self,
        sleep_session_no: int,
        deep_sleep_minutes: int,
        light_sleep_minutes: int,
        rem_sleep_minutes: int,
        deep_sleep_ratio: float,
        light_sleep_ratio: float,
        rem_sleep_ratio: float,
        session=None,
    ) -> None:
        std = session.get(SleepTimeDetail, int(sleep_session_no))
        if std is None:
            # create if missing
            std = SleepTimeDetail(
                sleep_session_no=int(sleep_session_no),
                deep_sleep_minutes=0,
                light_sleep_minutes=0,
                rem_sleep_minutes=0,
                deep_sleep_ratio=0.0,
                light_sleep_ratio=0.0,
                rem_sleep_ratio=0.0,
            )
            session.add(std)
        std.deep_sleep_minutes = int(deep_sleep_minutes)
        std.light_sleep_minutes = int(light_sleep_minutes)
        std.rem_sleep_minutes = int(rem_sleep_minutes)
        std.deep_sleep_ratio = float(deep_sleep_ratio)
        std.light_sleep_ratio = float(light_sleep_ratio)
        std.rem_sleep_ratio = float(rem_sleep_ratio)
        try:
            self._logger.info(
                "sleep_time_detail_updated",
                extra={
                    "session_no": int(sleep_session_no),
                    "deep_min": int(deep_sleep_minutes),
                    "light_min": int(light_sleep_minutes),
                    "rem_min": int(rem_sleep_minutes),
                    "deep_ratio": float(deep_sleep_ratio),
                    "light_ratio": float(light_sleep_ratio),
                    "rem_ratio": float(rem_sleep_ratio),
                },
            )
        except Exception:
            pass

    @session_scope
    def replace_analysis(
        self,
        sleep_session_no: int,
        details: list[tuple[str, str, Difficulty, Effect, list[tuple[int, str]]]],
        session=None,
    ) -> None:
        session.query(AnalysisStep).filter(AnalysisStep.analysis_detail_no.in_(
            session.query(AnalysisDetail.analysis_detail_no).filter(AnalysisDetail.sleep_session_no == int(sleep_session_no))
        )).delete(synchronize_session=False)
        session.query(AnalysisDetail).filter(AnalysisDetail.sleep_session_no == int(sleep_session_no)).delete(synchronize_session=False)

        for title, description, difficulty, effect, steps in details:
            ad = AnalysisDetail(
                sleep_session_no=int(sleep_session_no),
                title=str(title),
                description=str(description),
                difficulty=difficulty,
                effect=effect,
            )
            session.add(ad)
            session.flush()
            for step_index, content in steps:
                session.add(AnalysisStep(analysis_detail_no=int(ad.analysis_detail_no), step_index=int(step_index), content=str(content)))
        try:
            self._logger.info("analysis_details_replaced", extra={"session_no": int(sleep_session_no), "n_details": len(details)})
        except Exception:
            pass


