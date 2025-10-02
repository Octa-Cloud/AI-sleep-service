from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.api.domain.domain.entity.user_entity import User


class UserRepository(ABC):

    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[User]:
        raise NotImplementedError

    @abstractmethod
    def save(self, user: User) -> User:
        raise NotImplementedError
