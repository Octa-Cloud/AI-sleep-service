from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Tuple

from app.api.domain.domain.entity.periodic_report_entity import PeriodicReport


class PeriodicReportRepository(ABC):

    @abstractmethod
    def upsert(self, entity: PeriodicReport, points: List[Tuple[date, int]]) -> None:
        raise NotImplementedError


