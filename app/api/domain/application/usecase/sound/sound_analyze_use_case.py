from __future__ import annotations

from app.api.common.tsid import generate_int as generate_tsid_int


class SoundAnalyzeUseCase:
    def __init__(self, validator, producer, topic_input_raw: str) -> None:
        self._validator = validator
        self._producer = producer
        self._topic = topic_input_raw

    async def execute(self, sleep_session_no: int, sound_bytes: bytes) -> None:
        # Validate input (ffprobe check)
        if self._validator is not None:
            self._validator.validate(sound_bytes)

        # Publish protobuf message
        try:
            from app.common.kafka.dto import sound_pb2 as pb  # type: ignore
        except Exception as e:
            raise RuntimeError("Protobuf stubs not generated for sound. Run scripts/gen_protos.py") from e

        # 2) create trace id (TSID)
        tid = str(generate_tsid_int())

        msg = pb.SoundInputRaw(
            data=sound_bytes,
            trace_id=tid,
            session_no=int(sleep_session_no),
        )

        # 3) build protobuf message
        key = f"{int(sleep_session_no)}:{tid}"
        import time
        recorded_at_ms = int(time.time() * 1000)
        headers = {
            "trace_id": tid,
            "session_no": str(int(sleep_session_no)),
            "version": "1",
            "content-type": "application/x-protobuf;msg=SoundInputRaw",
            # define recorded_at at validation end time
            "recorded_at_ms": str(int(recorded_at_ms)),
        }
        getattr(self._producer, "send_bytes")(self._topic, key=key, value_bytes=msg.SerializeToString(), headers=headers)  # type: ignore[misc]


