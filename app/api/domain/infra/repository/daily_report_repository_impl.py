from __future__ import annotations

from datetime import date
from typing import List, Optional
from sqlalchemy.exc import IntegrityError

from sqlalchemy.orm import Session

from app.api.domain.domain.repository.daily_report_repository import DailyReportRepository
from app.api.domain.domain.vo.report_value_object import (
    DailyReportData,
    AnalysisDetailData,
    AnalysisStepData,
)
from app.api.domain.domain.entity.sleep_session_entity import (
    DailyReport as DailyReportEntity,
    SleepTimeDetail,
    AnalysisDetail,
    AnalysisStep,
)


class SqlAlchemyDailyReportRepository(DailyReportRepository):
    def __init__(self, session: Session) -> None:
        self._session = session

    def _row_to_vo(self, dr: DailyReportEntity, std: SleepTimeDetail | None, details: List[AnalysisDetail], steps_by_detail: dict[int, List[AnalysisStep]]) -> DailyReportData:
        detail_vos: List[AnalysisDetailData] = []
        for d in details:
            steps = [
                AnalysisStepData(step_index=int(s.step_index), content=str(s.content))
                for s in sorted(steps_by_detail.get(int(d.analysis_detail_no), []), key=lambda x: int(x.step_index))
            ]
            detail_vos.append(
                AnalysisDetailData(
                    title=str(d.title),
                    description=str(d.description),
                    difficulty=str(d.difficulty.value),
                    effect=str(d.effect.value),
                    steps=steps,
                )
            )

        return DailyReportData(
            sleep_session_no=int(dr.sleep_session_no),
            user_no=int(dr.user_no),
            sleep_date=dr.created_at.date(),  # 컷오프 적용은 서비스/유즈케이스에서
            created_at=dr.created_at,
            memo=dr.memo,
            score=int(getattr(dr, "score", 0) or 0),
            total_sleep_minutes=(
                int(std.deep_sleep_minutes + std.light_sleep_minutes + std.rem_sleep_minutes)
                if std is not None else None
            ),
            deep_sleep_minutes=int(std.deep_sleep_minutes) if std is not None else None,
            light_sleep_minutes=int(std.light_sleep_minutes) if std is not None else None,
            rem_sleep_minutes=int(std.rem_sleep_minutes) if std is not None else None,
            deep_sleep_ratio=float(std.deep_sleep_ratio) if std is not None else None,
            light_sleep_ratio=float(std.light_sleep_ratio) if std is not None else None,
            rem_sleep_ratio=float(std.rem_sleep_ratio) if std is not None else None,
            analysis_details=detail_vos,
        )

    def get_by_date(self, user_no: int, sleep_date: date) -> Optional[DailyReportData]:
        q = (
            self._session.query(DailyReportEntity)
            .filter(DailyReportEntity.user_no == int(user_no))
            .filter(DailyReportEntity.created_at >= date(sleep_date.year, sleep_date.month, sleep_date.day))
            .filter(DailyReportEntity.created_at < date(sleep_date.year, sleep_date.month, sleep_date.day).fromordinal(sleep_date.toordinal()+1))
        )
        dr = q.first()
        if dr is None:
            return None
        std = self._session.query(SleepTimeDetail).filter(SleepTimeDetail.sleep_session_no == int(dr.sleep_session_no)).first()
        details = self._session.query(AnalysisDetail).filter(AnalysisDetail.sleep_session_no == int(dr.sleep_session_no)).all()
        steps = self._session.query(AnalysisStep).filter(AnalysisStep.analysis_detail_no.in_([int(d.analysis_detail_no) for d in details] or [0])).all()
        steps_by_detail: dict[int, List[AnalysisStep]] = {}
        for s in steps:
            steps_by_detail.setdefault(int(s.analysis_detail_no), []).append(s)
        return self._row_to_vo(dr, std, details, steps_by_detail)

    def get_range(self, user_no: int, start_date: date, end_date: date) -> List[DailyReportData]:
        q = (
            self._session.query(DailyReportEntity)
            .filter(DailyReportEntity.user_no == int(user_no))
            .filter(DailyReportEntity.created_at >= start_date)
            .filter(DailyReportEntity.created_at <= end_date)
            .order_by(DailyReportEntity.created_at.asc())
        )
        out: List[DailyReportData] = []
        rows = q.all()
        if not rows:
            return out
        # Preload details/steps in batch
        session_nos = [int(r.sleep_session_no) for r in rows]
        std_map = {int(std.sleep_session_no): std for std in self._session.query(SleepTimeDetail).filter(SleepTimeDetail.sleep_session_no.in_(session_nos)).all()}
        details = self._session.query(AnalysisDetail).filter(AnalysisDetail.sleep_session_no.in_(session_nos)).all()
        steps = self._session.query(AnalysisStep).filter(AnalysisStep.analysis_detail_no.in_([int(d.analysis_detail_no) for d in details] or [0])).all()
        steps_by_detail: dict[int, List[AnalysisStep]] = {}
        for s in steps:
            steps_by_detail.setdefault(int(s.analysis_detail_no), []).append(s)
        # group details per session
        details_by_session: dict[int, List[AnalysisDetail]] = {}
        for d in details:
            details_by_session.setdefault(int(d.sleep_session_no), []).append(d)
        for r in rows:
            out.append(self._row_to_vo(r, std_map.get(int(r.sleep_session_no)), details_by_session.get(int(r.sleep_session_no), []), steps_by_detail))
        return out

    def upsert_daily_report(self, sleep_session_no: int, user_no: int, created_at, memo: str | None, score: int | None) -> None:
        dr = self._session.get(DailyReportEntity, int(sleep_session_no))
        if dr is None:
            dr = DailyReportEntity(
                sleep_session_no=int(sleep_session_no),
                user_no=int(user_no),
                created_at=created_at,
                memo=memo,
                score=score,
            )
            self._session.add(dr)
            self._session.flush()
            return
        if score is not None:
            dr.score = int(score)
        self._session.flush()


