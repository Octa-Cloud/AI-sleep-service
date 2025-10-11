from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Awaitable, Callable
from aiokafka import AIOKafkaConsumer


class AsyncKafkaConsumerRunner:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str, handler: Callable[[bytes, dict[str, str]], Awaitable[None] | None], dlq_topic: str | None = None, retry_max_attempts: int | None = None, retry_backoff_ms: int | None = None, started_event: asyncio.Event | None = None) -> None:
        self._bootstrap = bootstrap_servers
        self._topic = topic
        self._group = group_id
        self._handler = handler
        self._dlq_topic = dlq_topic
        self._retry_max_attempts = retry_max_attempts if retry_max_attempts is not None else int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
        self._retry_backoff_ms = retry_backoff_ms if retry_backoff_ms is not None else int(os.getenv("RETRY_BACKOFF_MS", "200"))
        
        # SASL/SSL configuration for Confluent Cloud
        consumer_config = {
            "bootstrap_servers": self._bootstrap,
            "group_id": self._group,
            "enable_auto_commit": False,
            "auto_offset_reset": "earliest",
        }
        
        # Add SASL authentication if configured
        security_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL")
        if security_protocol:
            consumer_config["security_protocol"] = security_protocol
            consumer_config["sasl_mechanism"] = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
            consumer_config["sasl_plain_username"] = os.getenv("KAFKA_SASL_USERNAME", "")
            consumer_config["sasl_plain_password"] = os.getenv("KAFKA_SASL_PASSWORD", "")
        
        self._consumer = AIOKafkaConsumer(
            self._topic,
            **consumer_config
        )
        self._task: asyncio.Task | None = None
        from app.common.kafka.producer import KafkaProducerClient
        self._producer = KafkaProducerClient(self._bootstrap)
        self._started_event = started_event

    async def start(self) -> None:
        await self._consumer.start()
        if self._started_event is not None:
            self._started_event.set()
        self._task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._consumer.stop()

    async def _consume_loop(self) -> None:
        try:
            while True:
                msg = await self._consumer.getone()
                try:
                    headers = {k: (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)) for k, v in (msg.headers or [])}
                    attempts = 0
                    while True:
                        try:
                            result = self._handler(msg.value or b"", headers)
                            if asyncio.iscoroutine(result):
                                await result
                            await self._consumer.commit()
                            break
                        except Exception as e:
                            attempts += 1
                            if attempts < self._retry_max_attempts:
                                await asyncio.sleep(self._retry_backoff_ms / 1000)
                                continue
                            # Send to DLQ if configured
                            if self._dlq_topic:
                                dlq_headers = dict(headers)
                                dlq_headers["content-type"] = "application/octet-stream;msg=DLQEnvelope"
                                key = headers.get("session_no", "") + ":" + headers.get("trace_id", "")
                                # Forward raw bytes to DLQ with original headers plus error info
                                dlq_headers["error"] = str(e)
                                self._producer.send_bytes(self._dlq_topic, key=key, value_bytes=(msg.value or b""), headers=dlq_headers)
                            await self._consumer.commit()
                            break
                except Exception:
                    # commit and continue on unexpected handler errors
                    await self._consumer.commit()
        except asyncio.CancelledError:
            return


