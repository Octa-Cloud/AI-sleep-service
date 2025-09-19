from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class MessageProducer(ABC):

    @abstractmethod
    async def send(self, topic: str, key: bytes | None, value: bytes) -> None:
        raise NotImplementedError
