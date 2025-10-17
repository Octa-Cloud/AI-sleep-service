from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Literal


@dataclass(frozen=True)
class DailyReportData:
    # 식별/메타
    sleep_session_no: int
    user_no: int
    sleep_date: date
    # 세션 시작 시각(세션 created_at과 동일해야 함)
    created_at: datetime
    memo: Optional[str] = None

    # 핵심 수치 지표
    score: Optional[float] = None
    total_sleep_minutes: Optional[int] = None

    deep_sleep_minutes: Optional[int] = None
    light_sleep_minutes: Optional[int] = None
    rem_sleep_minutes: Optional[int] = None

    deep_sleep_ratio: Optional[float] = None
    light_sleep_ratio: Optional[float] = None
    rem_sleep_ratio: Optional[float] = None

    # 분석 결과(타이틀/설명/난이도/효과 + 단계)
    analysis_details: List["AnalysisDetailData"] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisStepData:
    step_index: int
    content: str


@dataclass(frozen=True)
class AnalysisDetailData:
    title: str
    description: str
    difficulty: Literal["EASY", "MEDIUM", "HARD"]
    effect: Literal["LOW", "MEDIUM", "HIGH"]
    steps: List[AnalysisStepData] = field(default_factory=list)


@dataclass(frozen=True)
class ScorePredictionPointData:
    date_index: date
    score: int


@dataclass(frozen=True)
class PeriodicReportData:
    # 기본 식별
    user_no: int
    duration_type: Literal["WEEKLY", "MONTHLY"]
    period_started_at: date

    # 합계 지표(엔티티 정합)
    sleep_session_count: int
    total_score: int
    total_sleep_time_minutes: int
    total_bed_time_minutes: int
    total_deep_sleep_time_minutes: int
    total_light_sleep_time_minutes: int
    total_rem_sleep_time_minutes: int

    # 텍스트 분석
    improvement: Optional[str] = None
    weakness: Optional[str] = None
    recommendation: Optional[str] = None
    score_prediction_description: Optional[str] = None

    # 시계열 포인트
    score_prediction_points: List[ScorePredictionPointData] = field(default_factory=list)


