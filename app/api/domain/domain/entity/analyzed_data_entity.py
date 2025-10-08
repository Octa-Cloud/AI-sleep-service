from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import BigInteger, Column, DateTime, Enum as SAEnum, ForeignKey, CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.mysql import SMALLINT as MYSQL_SMALLINT

from app.api.domain.domain.entity.base import Base


class SoundEventType(str, Enum):
    SNORE = "SNORE"
    BABY_CRYING = "BABY_CRYING"
    COUGH = "COUGH"
    MOUTH_BREATHING = "MOUTH_BREATHING"
    ANIMAL_NOISE = "ANIMAL_NOISE"
    CAR_HORN = "CAR_HORN"


class SoundEvent(Base):
    __tablename__ = "analyzed_sound_events"

    analyzed_sound_event_no = Column(BigInteger, primary_key=True)
    sleep_session_no = Column(BigInteger, ForeignKey("sleep_sessions.sleep_session_no"), nullable=True)
    event = Column(SAEnum(SoundEventType, name="sound_event"), nullable=True)
    recorded_at = Column(DateTime(timezone=False), nullable=False)


class SleepLevel(Base):
    __tablename__ = "analyzed_sleep_levels"

    analyzed_sleep_level_no = Column(BigInteger, primary_key=True)
    sleep_session_no = Column(BigInteger, ForeignKey("sleep_sessions.sleep_session_no"), nullable=True)
    level = Column(MYSQL_SMALLINT(unsigned=True), nullable=True)
    recorded_at = Column(DateTime(timezone=False), nullable=False)

    __table_args__ = (
        CheckConstraint("level >= 0 and level <= 6", name="chk_analyzed_sleep_levels_level"),
        UniqueConstraint("sleep_session_no", "recorded_at", name="uq_analyzed_sleep_levels_session_time"),
    )


