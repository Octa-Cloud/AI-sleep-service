from __future__ import annotations

from typing import Callable, List

from app.api.domain.domain.entity.analyzed_data_entity import SoundEvent
from app.api.domain.infra.repository.sound_event_repository_impl import SqlAlchemySoundEventRepository
from app.api.domain.infra.db.session import session_scope


class SoundEventService:
    def __init__(self, repo_factory: Callable[..., SqlAlchemySoundEventRepository]) -> None:
        # repo_factory should accept a keyword arg: session
        self._repo_factory = repo_factory

    def save_events(self, entities: List[SoundEvent]) -> int:
        if not entities:
            return 0
        with session_scope() as session:
            try:
                repo = self._repo_factory(session=session)
            except TypeError:
                repo = self._repo_factory()
            return repo.save_bulk(entities)


