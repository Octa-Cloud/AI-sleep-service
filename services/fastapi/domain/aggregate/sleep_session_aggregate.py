from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
)
from __future__ import annotations
from datetime import datetime
from enum import Enum
from sqlalchemy.dialects.mysql import SMALLINT as MYSQL_SMALLINT
from services.fastapi.domain.aggregate.base import Base

class Difficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class Effect(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SleepSession(Base):
    __tablename__ = "sleep_sessions"

    sleep_session_no = Column(BigInteger, primary_key=True, autoincrement=True)
    user_no = Column(BigInteger, ForeignKey("users.user_no"), nullable=False)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)


class DailyReport(Base):
    __tablename__ = "daily_reports"

    sleep_session_no = Column(BigInteger, ForeignKey("sleep_sessions.sleep_session_no", ondelete="CASCADE"), primary_key=True)
    memo = Column(String(255), nullable=True)
    user_no = Column(BigInteger, ForeignKey("users.user_no"), nullable=False)
    created_at = Column(DateTime, nullable=False)


class SleepTimeDetail(Base):
    __tablename__ = "sleep_time_details"

    sleep_session_no = Column(BigInteger, ForeignKey("daily_reports.sleep_session_no", ondelete="CASCADE"), primary_key=True)

    deep_sleep_minutes = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)
    light_sleep_minutes = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)
    rem_sleep_minutes = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)

    deep_sleep_ratio = Column(Numeric(4, 1), nullable=False)
    light_sleep_ratio = Column(Numeric(4, 1), nullable=False)
    rem_sleep_ratio = Column(Numeric(4, 1), nullable=False)


class AnalysisDetail(Base):
    __tablename__ = "analysis_details"

    analysis_detail_no = Column(BigInteger, primary_key=True, autoincrement=True)
    sleep_session_no = Column(BigInteger, ForeignKey("daily_reports.sleep_session_no", ondelete="CASCADE"), nullable=False)

    title = Column(String(255), nullable=False)
    description = Column(String(255), nullable=False)
    difficulty = Column(SAEnum(Difficulty, name="difficulty"), nullable=False)
    effect = Column(SAEnum(Effect, name="effect"), nullable=False)


class AnalysisStep(Base):
    __tablename__ = "analysis_steps"

    analysis_step_no = Column(BigInteger, primary_key=True, autoincrement=True)
    analysis_detail_no = Column(BigInteger, ForeignKey("analysis_details.analysis_detail_no", ondelete="CASCADE"), nullable=False)

    step_index = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)
    content = Column(String(255), nullable=False)
