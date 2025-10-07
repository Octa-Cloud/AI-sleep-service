from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from aiokafka import AIOKafkaConsumer


class KafkaStageRunner:
    def __init__(
        self,
        consumer: AIOKafkaConsumer,
        start_producer: Callable[[], Awaitable[None]],
        stop_producer: Callable[[], Awaitable[None]],
        handle_message: Callable[[bytes], Awaitable[None] | None],
    ) -> None:
        self._consumer = consumer
        self._start_producer = start_producer
        self._stop_producer = stop_producer
        self._handle = handle_message

    async def run_forever(self) -> int:
        await self._consumer.start()
        await self._start_producer()
        try:
            while True:
                msg = await self._consumer.getone()
                try:
                    result = self._handle(msg.value or b"")
                    if asyncio.iscoroutine(result):
                        await result
                    await self._consumer.commit()
                except Exception:
                    await self._consumer.commit()
        finally:
            await self._consumer.stop()
            await self._stop_producer()
        return 0


