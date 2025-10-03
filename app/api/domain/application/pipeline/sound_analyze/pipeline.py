from __future__ import annotations

import asyncio
import logging
from typing import Optional


from app.api.domain.application.pipeline.sound_analyze.tasks import (
    SoundSplitTask,
    SoundAnalyzeTask,
    SoundSaveTask,
)


logger = logging.getLogger('sound-analyze-pipeline')

class SoundAnalyzePipeline:
    def __init__(self) -> None:
        # Placeholder until sound services are implemented
        pass

    def start(self, sleep_session_no: int, wav_bytes: bytes) -> None:
        task = asyncio.create_task(self._execute(sleep_session_no, wav_bytes))

        def _on_done(t: asyncio.Task) -> None:
            try:
                t.result()
            except Exception as exc:
                logger.exception("Sound analyze pipeline failed", exc_info=exc)

        task.add_done_callback(_on_done)

    async def _execute(self, sleep_session_no: int, wav_bytes: bytes) -> None:
        # TODO: wire actual logic when sound services are implemented
        return None


