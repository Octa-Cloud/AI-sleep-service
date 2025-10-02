from __future__ import annotations

from typing import List


class SoundSplitTask:
    async def execute(self, wav_bytes: bytes) -> List[bytes]:
        return [wav_bytes]


class SoundAnalyzeTask:
    async def execute(self, chunks: List[bytes]) -> List[int]:
        return []


class SoundSaveTask:
    async def execute(self, sleep_session_no: int, classes: List[int]) -> None:
        return None


