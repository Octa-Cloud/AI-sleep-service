from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class BrainwaveChunkData:
    data: np.ndarray  # shape: (channels=2, samples)
    start_at: datetime  # segment start time (UTC naive)
    sampling_rate_hz: float


@dataclass(frozen=True)
class SoundChunkData:
    data: bytes
    start_at: datetime  # segment start time (UTC aware)
    sampling_rate_hz: float


