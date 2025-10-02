from __future__ import annotations

from datetime import datetime

from app.api.domain.domain.entity.sleep_session_entity import SleepSession
from app.api.common.decorator.session_scope import session_scope
from app.api.common.exception.custom.session_exceptions import SleepSessionExistsApiException


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
            created_at=datetime.utcnow(),
            finished_at=None,
        )
        return repo.insert(entity)


