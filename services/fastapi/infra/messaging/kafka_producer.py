from __future__ import annotations

import asyncio
import os
from typing import Optional

from services.fastapi.domain.messaging.producer import MessageProducer

try:
    from aiokafka import AIOKafkaProducer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AIOKafkaProducer = None  # type: ignore


class KafkaMessageProducer(MessageProducer):

    def __init__(self, bootstrap_servers: Optional[str] = None) -> None:
        if AIOKafkaProducer is None:
            raise RuntimeError("aiokafka is not installed")
        self._bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
        self._producer: Optional[AIOKafkaProducer] = None
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        async with self._lock:
            if self._producer is None:
                self._producer = AIOKafkaProducer(bootstrap_servers=self._bootstrap_servers)
                await self._producer.start()

    async def send(self, topic: str, key: bytes | None, value: bytes) -> None:
        await self._ensure_started()
        assert self._producer is not None
        await self._producer.send_and_wait(topic, value=value, key=key)

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
