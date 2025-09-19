from __future__ import annotations

import os
from typing import Any

from dependency_injector import containers, providers

from services.fastapi.infra.db.session import SessionLocal
from services.fastapi.infra.repository.user_repository_impl import SqlAlchemyUserRepository
from services.fastapi.infra.repository.sleep_session_repository_impl import SqlAlchemySleepSessionRepository
from services.fastapi.infra.messaging.local_producer import LocalMessageProducer

try:
    from services.fastapi.infra.messaging.kafka_producer import KafkaMessageProducer  # optional
except Exception:  # pragma: no cover
    KafkaMessageProducer = None  # type: ignore


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    session_factory = providers.Factory(SessionLocal)

    user_repository = providers.Factory(
        SqlAlchemyUserRepository,
        session=providers.Dependency(instance_of=Any)
    )

    sleep_session_repository = providers.Factory(
        SqlAlchemySleepSessionRepository,
        session=providers.Dependency(instance_of=Any)
    )

    _queue_backend = os.getenv("QUEUE_BACKEND", "local").lower()
    if _queue_backend == "kafka" and KafkaMessageProducer is not None:
        message_producer = providers.Singleton(
            KafkaMessageProducer,
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        )
    else:
        message_producer = providers.Singleton(LocalMessageProducer)