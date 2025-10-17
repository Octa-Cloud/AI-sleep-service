from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.api.domain.domain.entity.analyzed_data_entity import SoundEvent
from app.api.domain.domain.repository.sound_event_repository import SoundEventRepository


class SqlAlchemySoundEventRepository(SoundEventRepository):

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_bulk(self, entities: List[SoundEvent]) -> int:
        if not entities:
            return 0
        # Insert with provided primary key and enum value
        rows = []
        for e in entities:
            rows.append({
                "analyzed_sound_event_no": e.analyzed_sound_event_no,
                "sleep_session_no": e.sleep_session_no,
                "event": (e.event.value if e.event is not None else None),
                "recorded_at": e.recorded_at,
            })
        table = SoundEvent.__table__
        stmt = mysql_insert(table).values(rows)
        self._session.execute(stmt)
        return len(rows)



