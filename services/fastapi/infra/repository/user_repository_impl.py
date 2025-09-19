from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from services.fastapi.domain.aggregate.user_aggregate import User
from services.fastapi.domain.repository.user_repository import UserRepository


class SqlAlchemyUserRepository(UserRepository):

    def __init__(self, session: Session) -> None:
        self._session = session

    def find_by_id(self, user_id: int) -> Optional[User]:
        return self._session.get(User, int(user_id))

    def save(self, user: User) -> User:
        existing = self._session.get(User, int(user.user_no))
        if existing is None:
            self._session.add(user)
            return user
        existing.name = user.name
        existing.nickname = user.nickname
        existing.email = user.email
        existing.password = user.password
        existing.gender = user.gender
        return existing
