from __future__ import annotations
# 이 파일은 수면 단계 저장 서비스가 레포지토리에 위임하여 벌크 저장하는지 검증합니다.

from datetime import datetime, timezone
from typing import List

from app.api.domain.application.service.brainwave.sleep_level_service import SleepLevelService
from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
from app.api.domain.domain.entity.sleep_session_entity import SleepSession
from app.api.domain.infra.db import session as db_session
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.api.domain.infra.repository.sleep_level_repository_impl import SqlAlchemySleepLevelRepository
from app.api.domain.infra.repository.sleep_session_repository_impl import SqlAlchemySleepSessionRepository


class FakeSleepLevelRepository:
    def __init__(self, session=None) -> None:
        self.saved_entities: List[SleepLevel] = []

    def save_bulk(self, entities: List[SleepLevel]) -> int:
        self.saved_entities.extend(entities)
        return len(entities)


def _new_session() -> Session:
    return db_session.SessionLocal()


def test_insert_bulk_persists_rows_to_database():
    # 1) 선행 조건: 수면 세션 생성 (FK 충족: user_no=1은 reset_db 시드)
    with _new_session() as db:
        session_repo = SqlAlchemySleepSessionRepository(session=db)
        sleep_session = SleepSession(user_no=1, created_at=datetime.now(timezone.utc), finished_at=None)
        session_repo.insert(sleep_session)
        db.commit()
        sleep_session_no = int(sleep_session.sleep_session_no)

    # 2) 서비스 생성 (실제 레포지토리 사용)
    service = SleepLevelService(
        session_repo_factory=lambda **kwargs: None,
        sleep_level_repo_factory=lambda **kwargs: SqlAlchemySleepLevelRepository(session=kwargs.get("session")),
    )

    # 3) 엔터티 준비 및 저장
    entities = [
        SleepLevel(analyzed_sleep_level_no=1, sleep_session_no=sleep_session_no, level=2, recorded_at=datetime.now(timezone.utc)),
        SleepLevel(analyzed_sleep_level_no=2, sleep_session_no=sleep_session_no, level=3, recorded_at=datetime.now(timezone.utc)),
    ]
    service.insert_bulk(entities)

    # 4) DB에서 실제로 저장되었는지 확인
    with _new_session() as db:
        rows = db.execute(
            select(SleepLevel).where(SleepLevel.sleep_session_no == sleep_session_no)
        ).scalars().all()
        assert len(rows) == 2


