from __future__ import annotations

from typing import Optional

from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from app.api.domain.domain.entity.sleep_session_entity import SleepSession
from app.api.domain.domain.repository.sleep_session_repository import SleepSessionRepository


class SqlAlchemySleepSessionRepository(SleepSessionRepository):

    def __init__(self, session: Session) -> None:
        self._session = session

    def find_by_id(self, session_id: int) -> Optional[SleepSession]:
        return self._session.get(SleepSession, int(session_id))

    def find_ongoing_by_user_no(self, user_no: int) -> Optional[SleepSession]:
        stmt = (
            select(SleepSession)
            .where(SleepSession.user_no == int(user_no))
            .where(SleepSession.finished_at.is_(None))
            .order_by(desc(SleepSession.created_at))
            .limit(1)
        )
        return self._session.execute(stmt).scalars().first()

    def insert(self, session_entity: SleepSession) -> SleepSession:
        self._session.add(session_entity)
        return session_entity
