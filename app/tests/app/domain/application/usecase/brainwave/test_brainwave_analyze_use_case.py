from __future__ import annotations
# 유즈케이스가 입력을 검증하고 Kafka로 protobuf 메시지를 발행하는지 테스트합니다.

import sys
import types
import pytest

from app.api.domain.application.usecase.brainwave.brainwave_analyze_use_case import BrainwaveAnalyzeUseCase


class AlwaysOkValidator:
    def __init__(self) -> None:
        self.validated = False

    def validate(self, edf_bytes: bytes) -> None:
        self.validated = True


class FakeProducer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    def send_bytes(self, topic: str, key: str, value_bytes: bytes, headers: dict[str, str] | None = None) -> None:  # type: ignore[override]
        self.calls.append((topic, key, value_bytes, headers or {}))


class _FakePbMsg:
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    def SerializeToString(self) -> bytes:
        return b"fake"


def _install_fake_pb(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pb = types.SimpleNamespace(BrainwaveInputRaw=_FakePbMsg)
    module_name = "app.common.kafka.dto.brainwave_pb2"
    # Ensure the protobuf module import resolves to our fake
    if module_name in sys.modules:
        monkeypatch.setitem(sys.modules, module_name, fake_pb)
    else:
        sys.modules[module_name] = fake_pb
    # Also override the already-imported module-level reference used by the use case
    mod_name_uc = "app.api.domain.application.usecase.brainwave.brainwave_analyze_use_case"
    if mod_name_uc in sys.modules:
        uc_mod = sys.modules[mod_name_uc]
        monkeypatch.setattr(uc_mod, "pb", fake_pb, raising=False)


@pytest.mark.asyncio
async def test_execute_validates_and_publishes_protobuf(monkeypatch: pytest.MonkeyPatch):
    _install_fake_pb(monkeypatch)
    validator = AlwaysOkValidator()
    producer = FakeProducer()
    usecase = BrainwaveAnalyzeUseCase(validator=validator, producer=producer, topic_input_raw="brainwave.input.raw")

    await usecase.execute(123, b"edf-bytes")

    assert validator.validated is True
    assert len(producer.calls) == 1
    topic, key, value_bytes, headers = producer.calls[0]
    assert topic == "brainwave.input.raw"
    assert value_bytes == b"fake"
    assert headers.get("content-type", "").startswith("application/x-protobuf")


class FailingValidator:
    def validate(self, edf_bytes: bytes) -> None:
        raise ValueError("invalid")


@pytest.mark.asyncio
async def test_execute_propagates_validator_error(monkeypatch: pytest.MonkeyPatch):
    _install_fake_pb(monkeypatch)
    usecase = BrainwaveAnalyzeUseCase(validator=FailingValidator(), producer=FakeProducer(), topic_input_raw="t")
    with pytest.raises(ValueError):
        await usecase.execute(1, b"bad")


