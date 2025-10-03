from __future__ import annotations
# 이 파일은 유즈케이스가 입력을 검증하고 파이프라인을 시작하는지 테스트합니다.

import pytest

from app.api.domain.application.usecase.brainwave.brainwave_analyze_use_case import BrainwaveAnalyzeUseCase


class AlwaysOkValidator:
    def __init__(self) -> None:
        self.validated = False

    def validate(self, edf_bytes: bytes) -> None:
        self.validated = True


class SpyBrainwavePipeline:
    def __init__(self) -> None:
        self.started = False
        self.args = None

    def start(self, sleep_session_no: int, edf_bytes: bytes) -> None:
        self.started = True
        self.args = (sleep_session_no, edf_bytes)


@pytest.mark.asyncio
async def test_execute_validates_and_starts_pipeline():
    validator = AlwaysOkValidator()
    pipeline = SpyBrainwavePipeline()
    usecase = BrainwaveAnalyzeUseCase(validator=validator, pipeline=pipeline)

    await usecase.execute(123, b"edf-bytes")

    assert validator.validated is True
    assert pipeline.started is True
    assert pipeline.args[0] == 123
    assert isinstance(pipeline.args[1], (bytes, bytearray))


@pytest.mark.asyncio
async def test_execute_calls_validator_once():
    validator = AlwaysOkValidator()
    pipeline = SpyBrainwavePipeline()
    usecase = BrainwaveAnalyzeUseCase(validator=validator, pipeline=pipeline)

    await usecase.execute(1, b"x")
    assert validator.validated is True


class FailingValidator:
    def validate(self, edf_bytes: bytes) -> None:
        raise ValueError("invalid")


@pytest.mark.asyncio
async def test_execute_propagates_validator_error():
    usecase = BrainwaveAnalyzeUseCase(validator=FailingValidator(), pipeline=SpyBrainwavePipeline())
    with pytest.raises(ValueError):
        await usecase.execute(1, b"bad")


class RecordingPipeline(SpyBrainwavePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.start_count = 0

    def start(self, sleep_session_no: int, edf_bytes: bytes) -> None:
        self.start_count += 1
        super().start(sleep_session_no, edf_bytes)


@pytest.mark.asyncio
async def test_execute_triggers_pipeline_once():
    validator = AlwaysOkValidator()
    pipeline = RecordingPipeline()
    usecase = BrainwaveAnalyzeUseCase(validator=validator, pipeline=pipeline)

    await usecase.execute(10, b"edf")
    assert pipeline.start_count == 1
    assert pipeline.args[0] == 10


@pytest.mark.asyncio
async def test_execute_accepts_bytes_like():
    validator = AlwaysOkValidator()
    pipeline = SpyBrainwavePipeline()
    usecase = BrainwaveAnalyzeUseCase(validator=validator, pipeline=pipeline)

    await usecase.execute(2, bytearray(b"edf"))
    assert pipeline.started is True


