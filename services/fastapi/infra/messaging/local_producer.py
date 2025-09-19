from __future__ import annotations

import logging

from services.fastapi.domain.messaging.producer import MessageProducer


logger = logging.getLogger(__name__)


class LocalMessageProducer(MessageProducer):

    async def send(self, topic: str, key: bytes | None, value: bytes) -> None:
        logger.info(f"[LOCAL PRODUCER] topic=%s key=%s value_len=%d", topic, key, len(value))
