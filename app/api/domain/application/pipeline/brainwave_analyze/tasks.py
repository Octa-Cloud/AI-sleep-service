from __future__ import annotations

from typing import List
import logging

from app.api.domain.application.service.brainwave.brainwave_chunk_splitter_service import BrainwaveChunkSplitterService
from app.api.domain.application.service.brainwave.brainwave_analyzer_service import BrainwaveAnalyzerService
from app.api.domain.application.service.brainwave.sleep_level_service import SleepLevelService
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData
from app.api.domain.domain.entity.analyzed_data_entity import SleepLevel
from app.api.domain.domain.vo.chunked_data_value_object import BrainwaveChunkData


class BrainwaveSplitTask:
    def __init__(self, splitter: BrainwaveChunkSplitterService) -> None:
        self._splitter = splitter
        self._logger = logging.getLogger('brainwave-analyze-split')

    async def execute(self, edf_bytes: bytes) -> List[BrainwaveChunkData]:
        self._logger.info(f"Splitting EDF bytes")
        return self._splitter.split(edf_bytes)


class BrainwaveAnalyzeTask:
    def __init__(self, analyzer: BrainwaveAnalyzerService) -> None:
        self._analyzer = analyzer
        self._logger = logging.getLogger('brainwave-analyze-analyze')

    async def execute(self, chunks: List[BrainwaveChunkData]) -> List[SleepLevelData]:
        # Analyzer now returns VO list directly
        self._logger.info(f"Analyzing {len(chunks)} chunks")
        return self._analyzer.analyze(chunks)


class BrainwaveSaveTask:
    def __init__(self, sleep_level_service: SleepLevelService) -> None:
        self._sleep_level_service = sleep_level_service
        self._logger = logging.getLogger('brainwave-analyze-save')

    async def execute(self, sleep_session_no: int, vo_list: List[SleepLevelData]) -> None:
        self._logger.info(f"Saving {len(vo_list)} sleep levels for session {sleep_session_no}")
        entities: List[SleepLevel] = self._sleep_level_service.data_to_entities(sleep_session_no, vo_list)
        self._sleep_level_service.insert_bulk(entities)


