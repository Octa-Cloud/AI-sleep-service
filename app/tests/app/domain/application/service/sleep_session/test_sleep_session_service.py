from __future__ import annotations
# 이 파일은 세션 시작/종료 서비스 동작을 검증합니다.

from datetime import datetime, timezone

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import select

import pytest

from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService
from app.api.domain.domain.entity.sleep_session_entity import SleepSession
from app.api.common.exception.custom.session_exceptions import SleepSessionExistsApiException
from app.api.domain.infra.repository.sleep_session_repository_impl import SqlAlchemySleepSessionRepository
from app.api.domain.infra import db as db_pkg
from app.api.domain.infra.db import session as db_session


class DummySession:
    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeSleepSessionRepository:
    def __init__(self, session=None, ongoing: SleepSession | None = None) -> None:
        self._ongoing = ongoing
        self._inserted: SleepSession | None = None

    def find_ongoing_by_user_no(self, user_no: int) -> SleepSession | None:
        return self._ongoing

    def insert(self, entity: SleepSession) -> None:
        self._inserted = entity


def test_begin_raises_when_ongoing_session_exists(monkeypatch):
    # 진행 중 세션이 있으면 begin()이 예외를 발생시켜야 합니다.
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())

    ongoing = SleepSession(sleep_session_no=1, user_no=1, created_at=datetime.now(timezone.utc), finished_at=None)
    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository(session=kwargs.get("session"), ongoing=ongoing))

    with pytest.raises(SleepSessionExistsApiException):
        service.begin(1)


def test_begin_inserts_when_no_ongoing_session(monkeypatch):
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())

    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository(session=kwargs.get("session")))
    saved = service.begin(99)

    assert isinstance(saved, SleepSession)
    assert saved.user_no == 99
    assert saved.finished_at is None


def test_begin_sets_created_at(monkeypatch):
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())
    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository(session=kwargs.get("session")))
    saved = service.begin(5)
    assert isinstance(saved.created_at, datetime)


def test_begin_with_different_users_independent(monkeypatch):
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())
    repo_a = FakeSleepSessionRepository()
    repo_b = FakeSleepSessionRepository()
    service_a = SleepSessionService(repo_factory=lambda **kwargs: repo_a)
    service_b = SleepSessionService(repo_factory=lambda **kwargs: repo_b)
    a = service_a.begin(1)
    b = service_b.begin(2)
    assert a.user_no == 1 and b.user_no == 2


def test_begin_raises_when_repo_reports_ongoing(monkeypatch):
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())
    ongoing = SleepSession(sleep_session_no=99, user_no=3, created_at=datetime.now(timezone.utc), finished_at=None)
    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository(ongoing=ongoing))
    with pytest.raises(SleepSessionExistsApiException):
        service.begin(3)


def test_get_current_sleep_session_no_returns_none(monkeypatch):
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())
    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository())
    no = service.get_current_sleep_session_no(1)
    assert no is None



def _new_session() -> Session:
    return db_session.SessionLocal()


def test_sleep_session_begin_and_finish_sets_finished_at():
    with _new_session() as db:
        session_repo = SqlAlchemySleepSessionRepository(session=db)
        service = SleepSessionService(repo_factory=lambda **kwargs: session_repo)

        # begin
        s = service.begin(user_no=1)
        db.commit()
        assert s.sleep_session_no is not None

        # finish
        service.finish(user_no=1)
        db.commit()

        # verify
        refreshed = db.execute(
            select(SleepSession).where(SleepSession.sleep_session_no == s.sleep_session_no)
        ).scalar_one()
        assert refreshed.finished_at is not None


