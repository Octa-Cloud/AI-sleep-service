from __future__ import annotations

from datetime import date
from enum import Enum

from sqlalchemy import BigInteger, Column, Date, Enum as SAEnum, SmallInteger, String, ForeignKey
from sqlalchemy.dialects.mysql import SMALLINT as MYSQL_SMALLINT

from app.api.domain.domain.entity.base import Base


class DurationType(str, Enum):
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"

class PeriodicReport(Base):
    __tablename__ = "periodic_reports"

    periodic_report_no = Column(BigInteger, primary_key=True, autoincrement=True)
    user_no = Column(BigInteger, ForeignKey("users.user_no"), nullable=False)

    sleep_session_count = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)

    duration_type = Column(SAEnum(DurationType, name="duration_type"), nullable=False)
    period_started_at = Column(Date, nullable=False)

    total_score = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)
    total_sleep_time = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)

    total_bed_time_minutes = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)
    total_deep_sleep_time_minutes = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)
    total_light_sleep_time_minutes = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)
    total_rem_sleep_time_minutes = Column(MYSQL_SMALLINT(unsigned=True), nullable=False)

    improvement = Column(String(500))
    weakness = Column(String(500))
    recommendation = Column(String(500))
    score_prediction_description = Column(String(500))


class ScorePredictionPoint(Base):
    __tablename__ = "score_prediction_points"

    score_prediction_point_no = Column(BigInteger, primary_key=True, autoincrement=True)
    periodic_report_no = Column(BigInteger, ForeignKey("periodic_reports.periodic_report_no", ondelete="CASCADE"), nullable=False)
    date_index = Column(Date, nullable=False)
    score = Column(SmallInteger, nullable=False)
