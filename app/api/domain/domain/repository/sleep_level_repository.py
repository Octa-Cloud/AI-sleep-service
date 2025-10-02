from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel


class SleepLevelRepository(ABC):

    @abstractmethod
    def save_bulk(self, entities: List[SleepLevel]) -> int:
        raise NotImplementedError


