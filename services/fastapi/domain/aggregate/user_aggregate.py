from __future__ import annotations

from enum import Enum
from typing import Optional

from sqlalchemy import BigInteger, Column, Enum as SAEnum, String

from services.fastapi.domain.aggregate.base import Base


class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"


class User(Base):
    __tablename__ = "users"

    user_no = Column(BigInteger, primary_key=True)
    name = Column(String(64), nullable=True)
    nickname = Column(String(64), nullable=True)
    email = Column(String(128), nullable=False)
    password = Column(String(64), nullable=False)
    gender = Column(SAEnum(Gender, name="gender"), nullable=True)
