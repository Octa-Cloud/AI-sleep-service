from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from app.api.domain.domain.entity.user_entity import User
from app.api.domain.domain.repository.user_repository import UserRepository


class SqlAlchemyUserRepository(UserRepository):

    def __init__(self, session: Session) -> None:
        self._session = session

    def find_by_id(self, user_id: int) -> Optional[User]:
        return self._session.get(User, int(user_id))
