from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.api.domain.domain.entity.analyzed_data_entity import SoundEventType


@dataclass(frozen=True)
class SoundEventData:
    event: SoundEventType
    recorded_at: datetime


@dataclass(frozen=True)
class SleepLevelData:
    level: int
    recorded_at: datetime

