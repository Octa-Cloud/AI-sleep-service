from __future__ import annotations

from typing import Protocol


class KafkaMessageHandler(Protocol):
    """Domain-level handler invoked for each Kafka record value.

    Handlers are designed to be stateful per-topic-consumer if needed (e.g., aggregation buffers).
    They must be callable with raw bytes and decoded headers.
    """

    def __call__(self, value: bytes, headers: dict[str, str]) -> None: ...


