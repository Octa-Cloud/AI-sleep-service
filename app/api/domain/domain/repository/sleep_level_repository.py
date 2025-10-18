from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData


class SleepLevelRepository(ABC):

    @abstractmethod
    def save_bulk(self, entities: List[SleepLevel]) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_by_session(self, sleep_session_no: int) -> List[SleepLevelData]:
        raise NotImplementedError

