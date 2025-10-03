from __future__ import annotations
# 이 파일은 분석 태스크가 분석 결과를 수면 단계 VO로 변환하는지 검증합니다.

from datetime import datetime, timezone
from typing import List

import numpy as np
import pytest

from app.api.domain.application.pipeline.brainwave_analyze.tasks import BrainwaveAnalyzeTask
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData
from app.api.domain.domain.vo.chunked_data_value_object import BrainwaveChunkData


class _FakeAnalyzer:
    def __init__(self, classes: List[int]) -> None:
        self._classes = classes

    def analyze(self, chunk_data_list: List[BrainwaveChunkData]) -> List[SleepLevelData]:
        # Map each class to SleepLevelData with start_at timestamp
        out: List[SleepLevelData] = []
        for chunk in chunk_data_list:
            start_at = chunk.start_at
            for cls in self._classes:
                out.append(SleepLevelData(level=int(cls), recorded_at=start_at))
        return out


@pytest.mark.asyncio
async def test_analyze_task_maps_classes_to_sleep_level_data_with_timestamps():
    sampling_rate = 100.0
    # 1 epoch = 30 seconds => 3000 samples; make 1 epoch with two channels
    data = np.zeros((2, int(30 * sampling_rate)), dtype=np.float32)
    start_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    chunk = BrainwaveChunkData(data=data, start_at=start_at, sampling_rate_hz=sampling_rate)

    analyzer = _FakeAnalyzer(classes=[4])
    task = BrainwaveAnalyzeTask(analyzer)

    result = await task.execute([chunk])
    assert len(result) == 1
    assert result[0].level == 4
    # recorded_at should equal chunk.start_at for first epoch
    assert result[0].recorded_at == start_at


