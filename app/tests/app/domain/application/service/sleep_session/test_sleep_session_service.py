from __future__ import annotations
# 이 파일은 세션 시작 서비스가 중복 세션을 막고, 신규 생성 시 올바른 값을 반환하는지 테스트합니다.

from datetime import datetime, timezone

import pytest

from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService
from app.api.domain.domain.entity.sleep_session_entity import SleepSession
from app.api.common.exception.custom.session_exceptions import SleepSessionExistsApiException
from app.api.domain.infra import db as db_pkg


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
    # Make @session_scope use a dummy session (no real DB)
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())

    ongoing = SleepSession(sleep_session_no=1, user_no=1, created_at=datetime.now(timezone.utc), finished_at=None)
    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository(session=kwargs.get("session"), ongoing=ongoing))

    with pytest.raises(SleepSessionExistsApiException):
        service.begin(1)


def test_begin_inserts_when_no_ongoing_session(monkeypatch):
    # 진행 중 세션이 없으면 신규 세션을 생성해야 합니다.
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())

    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository(session=kwargs.get("session")))
    saved = service.begin(99)

    assert isinstance(saved, SleepSession)
    assert saved.user_no == 99
    assert saved.finished_at is None


def test_begin_sets_created_at(monkeypatch):
    # 생성된 세션에는 생성 시간이 설정되어 있어야 합니다.
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())
    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository(session=kwargs.get("session")))
    saved = service.begin(5)
    assert isinstance(saved.created_at, datetime)


def test_begin_with_different_users_independent(monkeypatch):
    # 서로 다른 사용자로 호출하면 각각 독립된 세션이 생성되어야 합니다.
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())
    repo_a = FakeSleepSessionRepository()
    repo_b = FakeSleepSessionRepository()
    service_a = SleepSessionService(repo_factory=lambda **kwargs: repo_a)
    service_b = SleepSessionService(repo_factory=lambda **kwargs: repo_b)
    a = service_a.begin(1)
    b = service_b.begin(2)
    assert a.user_no == 1 and b.user_no == 2


def test_begin_raises_when_repo_reports_ongoing(monkeypatch):
    # 레포지토리에서 진행 중 세션을 반환하면 예외가 발생해야 합니다.
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())
    ongoing = SleepSession(sleep_session_no=99, user_no=3, created_at=datetime.now(timezone.utc), finished_at=None)
    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository(ongoing=ongoing))
    with pytest.raises(SleepSessionExistsApiException):
        service.begin(3)


def test_get_current_sleep_session_no_returns_none(monkeypatch):
    # 진행 중 세션이 없으면 현재 세션 번호 조회는 None을 반환해야 합니다.
    monkeypatch.setattr(db_pkg.session, "SessionLocal", lambda: DummySession())
    service = SleepSessionService(repo_factory=lambda **kwargs: FakeSleepSessionRepository())
    no = service.get_current_sleep_session_no(1)
    assert no is None


