from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.api.domain.domain.entity.sleep_session_entity import SleepSession


class SleepSessionRepository(ABC):

    @abstractmethod
    def find_by_id(self, session_id: int) -> Optional[SleepSession]:
        raise NotImplementedError

    @abstractmethod
    def find_ongoing_by_user_no(self, user_no: int) -> Optional[SleepSession]:
        raise NotImplementedError

    @abstractmethod
    def insert(self, session: SleepSession) -> None:
        raise NotImplementedError
