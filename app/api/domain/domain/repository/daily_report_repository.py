from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

from app.api.domain.domain.vo.report_value_object import DailyReportData


class DailyReportRepository(ABC):

    @abstractmethod
    def get_by_date(self, user_no: int, sleep_date: date) -> Optional[DailyReportData]:
        raise NotImplementedError

    @abstractmethod
    def get_range(self, user_no: int, start_date: date, end_date: date) -> List[DailyReportData]:
        raise NotImplementedError

    @abstractmethod
    def upsert_daily_report(self, sleep_session_no: int, user_no: int, created_at, memo: str | None, score: int | None) -> None:
        raise NotImplementedError


