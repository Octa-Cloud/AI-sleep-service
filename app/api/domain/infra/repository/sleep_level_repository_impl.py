from __future__ import annotations

from typing import Iterable, List

from sqlalchemy.orm import Session
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
from app.api.domain.domain.repository.sleep_level_repository import SleepLevelRepository
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData


class SqlAlchemySleepLevelRepository(SleepLevelRepository):

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_bulk(self, entities: List[SleepLevel]) -> int:
        if not entities:
            return 0
        # Convert ORM objects to dict rows
        rows = [
            {
                "analyzed_sleep_level_no": e.analyzed_sleep_level_no,
                "sleep_session_no": e.sleep_session_no,
                "level": e.level,
                "recorded_at": e.recorded_at,
            }
            for e in entities
        ]
        table = SleepLevel.__table__
        stmt = mysql_insert(table).values(rows)
        # Idempotent upsert keyed by unique (sleep_session_no, recorded_at)
        upsert_stmt = stmt.on_duplicate_key_update(
            level=stmt.inserted.level,  # update to latest level if reprocessed
            recorded_at=stmt.inserted.recorded_at,
        )
        self._session.execute(upsert_stmt)
        return len(rows)


