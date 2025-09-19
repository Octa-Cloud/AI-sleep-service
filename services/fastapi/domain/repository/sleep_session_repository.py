from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Iterable

from services.fastapi.domain.aggregate.sleep_session_aggregate import SleepSession


class SleepSessionRepository(ABC):

    @abstractmethod
    def find_by_id(self, session_id: int) -> Optional[SleepSession]:
        raise NotImplementedError

    @abstractmethod
    def find_latest_by_user(self, user_id: int) -> Optional[SleepSession]:
        raise NotImplementedError

    @abstractmethod
    def find_all_by_user(self, user_id: int) -> Iterable[SleepSession]:
        raise NotImplementedError

    @abstractmethod
    def save(self, session: SleepSession) -> SleepSession:
        raise NotImplementedError
