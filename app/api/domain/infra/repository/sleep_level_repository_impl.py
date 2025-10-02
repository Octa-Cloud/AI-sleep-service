from __future__ import annotations

from typing import Iterable, List

from sqlalchemy.orm import Session

from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
from app.api.domain.domain.repository.sleep_level_repository import SleepLevelRepository
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData


class SqlAlchemySleepLevelRepository(SleepLevelRepository):

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_bulk(self, entities: List[SleepLevel]) -> int:
        self._session.bulk_save_objects(entities)
        return len(entities)


