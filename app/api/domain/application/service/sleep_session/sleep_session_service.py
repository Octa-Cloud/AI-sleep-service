from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.api.domain.domain.entity.sleep_session_entity import SleepSession
from app.api.common.decorator.session_scope import session_scope
from app.api.common.exception.custom.session_exceptions import SleepSessionExistsApiException
from app.api.common.exception.custom.session_exceptions import SleepSessionNotFoundApiException


class SleepSessionService:
    def __init__(self, repo_factory) -> None:
        self._repo_factory = repo_factory

    @session_scope
    def begin(self, user_no: int, session=None) -> SleepSession:
        repo = self._repo_factory(session=session)

        ongoing = repo.find_ongoing_by_user_no(int(user_no))
        if ongoing is not None:
            raise SleepSessionExistsApiException()

        entity = SleepSession(
            user_no=int(user_no),
            created_at=datetime.now(timezone.utc),
            finished_at=None,
        )
        repo.insert(entity)
        return entity

    @session_scope
    def finish(self, user_no: int, session=None) -> None:
        repo = self._repo_factory(session=session)

        ongoing = repo.find_ongoing_by_user_no(int(user_no))
        if ongoing is None:
            raise SleepSessionNotFoundApiException()

        ongoing.finished_at = datetime.now(timezone.utc)

    @session_scope
    def get_current_sleep_session_no(self, user_no: int, session=None) -> int:
        repo = self._repo_factory(session=session)
        current_sleep_session =  repo.find_ongoing_by_user_no(int(user_no))
        if current_sleep_session is None:
            return None
        return current_sleep_session.sleep_session_no

