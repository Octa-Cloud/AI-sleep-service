from __future__ import annotations

import asyncio

from app.api.domain.application.pipeline.brainwave_analyze.pipeline import BrainwaveAnalyzePipeline
from app.api.domain.application.service.brainwave.brainwave_data_validator_service import BrainwaveDataValidatorService


class BrainwaveAnalyzeUseCase:
    def __init__(
        self,
        validator: BrainwaveDataValidatorService,
        pipeline: BrainwaveAnalyzePipeline,
    ) -> None:
        self._validator = validator
        self._pipeline = pipeline

    async def execute(self, sleep_session_no: int, edf_bytes: bytes) -> None:
        self._validator.validate(edf_bytes) # sync
        self._pipeline.start(sleep_session_no, edf_bytes)  # async and return

