from __future__ import annotations
# 이 파일은 분석 태스크가 분석 결과를 수면 단계 VO로 변환하는지 검증합니다.

from datetime import datetime, timezone
from typing import List

import numpy as np
import pytest

from app.api.domain.application.service.brainwave.brainwave_analyzer_service import BrainwaveAnalyzerService
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData
from app.api.domain.domain.vo.chunked_data_value_object import BrainwaveChunkData


class _FakeAnalyzerService(BrainwaveAnalyzerService):
    def __init__(self) -> None:  # type: ignore[override]
        pass

    def analyze(self, chunks: List[BrainwaveChunkData]) -> List[SleepLevelData]:  # type: ignore[override]
        out: List[SleepLevelData] = []
        for c in chunks:
            out.append(SleepLevelData(level=4, recorded_at=c.start_at))
        return out


@pytest.mark.asyncio
async def test_analyzer_service_maps_classes_to_sleep_level_data_with_timestamps():
    sampling_rate = 100.0
    data = np.zeros((2, int(30 * sampling_rate)), dtype=np.float32)
    start_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    chunk = BrainwaveChunkData(data=data, start_at=start_at, sampling_rate_hz=sampling_rate)

    analyzer = _FakeAnalyzerService()
    result = analyzer.analyze([chunk])
    assert len(result) == 1
    assert result[0].level == 4
    assert result[0].recorded_at == start_at


