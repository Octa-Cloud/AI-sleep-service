from __future__ import annotations

import asyncio
from typing import List

from app.common import config
from app.common.kafka.consumer import AsyncKafkaConsumerRunner
from app.api.domain.consumer.brainwave.aggregator_handler import BrainwaveAggregatorHandler
from app.api.domain.consumer.brainwave.db_writer_handler import BrainwaveDbWriterHandler


class KafkaSubscriptionsFactory:
    def __init__(self, container) -> None:
        self._container = container

    def create_all(self) -> List[AsyncKafkaConsumerRunner]:
        brokers = config.KAFKA_BROKERS
        dlq_topic = config.TOPIC_DLQ

        producer = getattr(self._container, "kafka_producer", None)
        agg_handler = BrainwaveAggregatorHandler(producer)
        db_writer_service = getattr(self._container, "brainwave_sleeplevel")
        db_handler = BrainwaveDbWriterHandler(db_writer_service, use_protobuf=config.KAFKA_PROTOBUF_ENABLED)

        agg_started = asyncio.Event()
        db_started = asyncio.Event()

        analyzed_topic = config.TOPIC_BRAINWAVE_ANALYZED_EPOCH
        agg_group = config.GROUP_BRAINWAVE_AGGREGATOR
        persist_topic = config.TOPIC_BRAINWAVE_PERSIST_REQUESTS
        db_group = config.GROUP_BRAINWAVE_DB_WRITER

        consumers = [
            AsyncKafkaConsumerRunner(brokers, analyzed_topic, agg_group, handler=agg_handler, dlq_topic=dlq_topic, started_event=agg_started),
            AsyncKafkaConsumerRunner(brokers, persist_topic, db_group, handler=db_handler, dlq_topic=dlq_topic, started_event=db_started),
        ]

        # store for orchestrator readiness checks
        self._started_events = [agg_started, db_started]
        return consumers

    def get_started_events(self) -> list[asyncio.Event]:
        return getattr(self, "_started_events", [])


