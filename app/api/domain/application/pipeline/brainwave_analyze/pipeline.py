from __future__ import annotations

import asyncio
import logging


from app.api.domain.application.pipeline.brainwave_analyze.tasks import (
    BrainwaveSplitTask,
    BrainwaveAnalyzeTask,
    BrainwaveSaveTask,
)
from app.api.domain.application.service.brainwave.brainwave_chunk_splitter_service import BrainwaveChunkSplitterService
from app.api.domain.application.service.brainwave.brainwave_analyzer_service import BrainwaveAnalyzerService
from app.api.domain.application.service.brainwave.sleep_level_service import SleepLevelService
from app.api.common.exception.custom.brainwave_exceptions import BrainwaveAnalyzeFailApiException
from typing import Any


logger = logging.getLogger('brainwave-analyze-pipeline')


class BrainwaveAnalyzePipeline:
    def __init__(
        self,
        splitter: BrainwaveChunkSplitterService,
        analyzer: BrainwaveAnalyzerService,
        sleep_level_service: SleepLevelService,
    ) -> None:
        self._split_task = BrainwaveSplitTask(splitter)
        self._analyze_task = BrainwaveAnalyzeTask(analyzer)
        self._save_task = BrainwaveSaveTask(sleep_level_service)

    def start(self, sleep_session_no: int, edf_bytes: bytes) -> None:
        task = asyncio.create_task(self._execute(sleep_session_no, edf_bytes))

        def _on_done(t: asyncio.Task) -> None:
            try:
                t.result()
            except Exception as exc:
                logger.exception("Brainwave analyze pipeline failed", exc_info=exc)

        task.add_done_callback(_on_done)

    async def _execute(self, sleep_session_no: int, edf_bytes: bytes) -> None:
        try:
            chunks = await self._split_task.execute(edf_bytes)
            classes = await self._analyze_task.execute(chunks)
            await self._save_task.execute(sleep_session_no, classes)
        except BrainwaveAnalyzeFailApiException:
            raise
        except Exception as exc:
            raise BrainwaveAnalyzeFailApiException() from exc


