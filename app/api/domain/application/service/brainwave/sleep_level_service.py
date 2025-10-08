from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List

from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
from app.api.domain.domain.entity.sleep_session_entity import SleepSession
from app.api.common.decorator.session_scope import session_scope
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData
from app.api.domain.domain.repository.sleep_level_repository import SleepLevelRepository
from app.api.common.tsid import generate_int as generate_tsid_int


class SleepLevelService:
    def __init__(
        self,
        session_repo_factory,
        sleep_level_repo_factory,
    ) -> None:
        self._session_repo_factory = session_repo_factory
        self._sleep_level_repo_factory = sleep_level_repo_factory

    def data_to_entities(self, sleep_session_no: int, vo_list: Iterable[SleepLevelData]) -> List[SleepLevel]:
        return [
            SleepLevel(
                analyzed_sleep_level_no=int(generate_tsid_int()),
                sleep_session_no=int(sleep_session_no),
                level=int(vo.level),
                recorded_at=vo.recorded_at,
            )
            for vo in vo_list
        ]

    @session_scope
    def insert_bulk(self, entities: List[SleepLevel], session=None) -> None:
        if not entities:
            return
        sleep_level_repo: SleepLevelRepository = self._sleep_level_repo_factory(session=session)
        sleep_level_repo.save_bulk(entities)


