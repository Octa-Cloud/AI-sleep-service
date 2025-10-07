from __future__ import annotations

from typing import Protocol

from app.api.domain.application.service.brainwave.brainwave_data_validator_service import BrainwaveDataValidatorService
from app.api.common.tsid import generate_int as generate_tsid_int
from app.common import config

try:
    from app.common.kafka.dto import brainwave_pb2 as pb  # type: ignore
except Exception:  # pragma: no cover
    pb = None  # type: ignore


class _Producer(Protocol):
    def send(self, topic: str, key: str, value: dict, headers: dict[str, str] | None = None) -> None: ...


class BrainwaveAnalyzeUseCase:
    def __init__(
        self,
        validator: BrainwaveDataValidatorService,
        producer: _Producer,
        topic_input_raw: str,
    ) -> None:
        self._validator = validator
        self._producer = producer
        self._topic_input_raw = topic_input_raw

    async def execute(self, sleep_session_no: int, edf_bytes: bytes) -> None:
        # 1) validate input synchronously
        self._validator.validate(edf_bytes)

        # 2) create trace id (TSID)
        trace_id = str(generate_tsid_int())

        # 3) build protobuf message (protobuf-only)
        key = f"{sleep_session_no}:{trace_id}"
        headers = {"trace_id": trace_id, "session_no": str(int(sleep_session_no)), "version": "1", "content-type": "application/x-protobuf;msg=BrainwaveInputRaw"}

        if pb is None:
            raise RuntimeError("Protobuf stubs not generated. Run scripts/gen_protos.sh")
        obj = pb.BrainwaveInputRaw(  # type: ignore[attr-defined]
            trace_id=trace_id,
            session_no=int(sleep_session_no),
            epoch_seconds=30,
            inline_bytes=edf_bytes,
        )
        if not hasattr(self._producer, "send_bytes"):
            raise RuntimeError("Producer does not support send_bytes; protobuf-only path requires it.")
        getattr(self._producer, "send_bytes")(self._topic_input_raw, key=key, value_bytes=obj.SerializeToString(), headers=headers)  # type: ignore[misc]

