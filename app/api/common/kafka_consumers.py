from __future__ import annotations

import asyncio
from typing import List

from app.api.common.kafka_subscriptions import KafkaSubscriptionsFactory


class KafkaConsumerOrchestrator:
    def __init__(self, container=None) -> None:
        self._container = container
        self._instances: list = []
        self._started_events: list[asyncio.Event] = []

    def start_all(self) -> None:
        factory = KafkaSubscriptionsFactory(self._container)
        self._instances = factory.create_all()
        self._started_events = factory.get_started_events()
        loop = asyncio.get_event_loop()
        for c in self._instances:
            loop.create_task(c.start())

    async def wait_ready(self) -> None:
        if not self._started_events:
            return
        for ev in self._started_events:
            await ev.wait()

    def stop_all(self) -> None:
        loop = asyncio.get_event_loop()
        for c in self._instances:
            loop.create_task(c.stop())
        self._instances.clear()


