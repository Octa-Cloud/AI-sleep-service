from __future__ import annotations

import json
from typing import Any
import asyncio
import json
from aiokafka import AIOKafkaProducer


class KafkaProducerClient:
    def __init__(self, bootstrap_servers: str) -> None:
        self._bootstrap = bootstrap_servers
        self._loop = asyncio.get_event_loop()
        
        # SASL/SSL configuration for Confluent Cloud
        producer_config = {
            "bootstrap_servers": self._bootstrap,
            "loop": self._loop
        }
        
        # Add SASL authentication if configured
        security_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL")
        if security_protocol:
            producer_config["security_protocol"] = security_protocol
            producer_config["sasl_mechanism"] = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
            producer_config["sasl_plain_username"] = os.getenv("KAFKA_SASL_USERNAME", "")
            producer_config["sasl_plain_password"] = os.getenv("KAFKA_SASL_PASSWORD", "")
        
        self._producer = AIOKafkaProducer(**producer_config)
        if not self._producer._closed:  # type: ignore[attr-defined]
            pass

    async def start(self) -> None:
        await self._producer.start()

    async def stop(self) -> None:
        try:
            await self._producer.stop()
        except Exception:
            pass

    def send(self, topic: str, key: str, value: dict, headers: dict[str, str] | None = None) -> None:
        # Fire-and-forget JSON publish
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        hdrs = [(k, v.encode("utf-8")) for k, v in (headers or {}).items()]
        async def _produce():
            await self._producer.send_and_wait(topic, value=data, key=key.encode("utf-8"), headers=hdrs)  # type: ignore[arg-type]
        self._loop.create_task(_produce())

    def send_bytes(self, topic: str, key: str, value_bytes: bytes, headers: dict[str, str] | None = None) -> None:
        # Fire-and-forget bytes publish (e.g., protobuf)
        hdrs = [(k, v.encode("utf-8")) for k, v in (headers or {}).items()]
        async def _produce():
            try:
                await self._producer.send_and_wait(topic, value=value_bytes, key=key.encode("utf-8"), headers=hdrs)  # type: ignore[arg-type]
            except Exception:
                # swallow send error; upstream will retry via consumer DLQ or aggregation loop
                pass
        self._loop.create_task(_produce())


