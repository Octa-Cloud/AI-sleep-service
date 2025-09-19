from __future__ import annotations

from typing import Iterable, Optional, List

from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from services.fastapi.domain.aggregate.sleep_session_aggregate import SleepSession
from services.fastapi.domain.repository.sleep_session_repository import SleepSessionRepository


class SqlAlchemySleepSessionRepository(SleepSessionRepository):

    def __init__(self, session: Session) -> None:
        self._session = session

    def find_by_id(self, session_id: int) -> Optional[SleepSession]:
        return self._session.get(SleepSession, int(session_id))

    def find_latest_by_user(self, user_id: int) -> Optional[SleepSession]:
        stmt = (
            select(SleepSession)
            .where(SleepSession.user_no == int(user_id))
            .order_by(desc(SleepSession.created_at))
            .limit(1)
        )
        return self._session.execute(stmt).scalars().first()

    def find_all_by_user(self, user_id: int) -> Iterable[SleepSession]:
        stmt = (
            select(SleepSession)
            .where(SleepSession.user_no == int(user_id))
            .order_by(desc(SleepSession.created_at))
        )
        rows: List[SleepSession] = list(self._session.execute(stmt).scalars().all())
        return rows

    def save(self, session_entity: SleepSession) -> SleepSession:
        existing = self._session.get(SleepSession, int(session_entity.sleep_session_no))
        if existing is None:
            self._session.add(session_entity)
            return session_entity
        existing.user_no = int(session_entity.user_no)
        existing.finished_at = session_entity.finished_at
        existing.created_at = session_entity.created_at
        return existing
