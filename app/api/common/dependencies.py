from __future__ import annotations

import os
from typing import Callable

from app.api.domain.infra.db.session import SessionLocal
from app.api.domain.infra.repository.sleep_session_repository_impl import SqlAlchemySleepSessionRepository
from app.api.domain.infra.repository.sleep_level_repository_impl import SqlAlchemySleepLevelRepository
from app.api.domain.application.service.brainwave.brainwave_data_validator_service import BrainwaveDataValidatorService
from app.api.domain.application.service.brainwave.brainwave_analyzer_service import BrainwaveAnalyzerService
from app.api.domain.application.service.brainwave.brainwave_chunk_splitter_service import BrainwaveChunkSplitterService
from app.api.domain.application.service.brainwave.sleep_level_service import SleepLevelService
from app.api.domain.application.usecase.brainwave.brainwave_analyze_use_case import BrainwaveAnalyzeUseCase
from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService


class Container:

    def __init__(self) -> None:
        # Repository factories (with session injection)
        self.session_factory: Callable[[], object] = SessionLocal
        self.sleep_session_repository_factory: Callable[..., SqlAlchemySleepSessionRepository] = (
            lambda *, session: SqlAlchemySleepSessionRepository(session=session)
        )
        self.sleep_level_repository_factory: Callable[..., SqlAlchemySleepLevelRepository] = (
            lambda *, session: SqlAlchemySleepLevelRepository(session=session)
        )

        # Services
        self.brainwave_validator = BrainwaveDataValidatorService()
        self.brainwave_splitter = BrainwaveChunkSplitterService()
        self.brainwave_analyzer = BrainwaveAnalyzerService(app_root=os.getcwd())

        self.brainwave_sleeplevel = SleepLevelService(
            session_repo_factory=self.sleep_session_repository_factory,
            sleep_level_repo_factory=self.sleep_level_repository_factory,
        )

        # Kafka producer client (to be implemented in app/common/kafka)
        try:
            from app.common.kafka.producer import KafkaProducerClient  # type: ignore
            kafka_bootstrap = os.getenv("KAFKA_BROKERS", "localhost:9092")
            self.kafka_producer = KafkaProducerClient(bootstrap_servers=kafka_bootstrap)
        except Exception:
            self.kafka_producer = None  # type: ignore
        self.topic_brainwave_input_raw = os.getenv("TOPIC_BRAINWAVE_INPUT_RAW", "brainwave.input.raw")

        # Factories for higher-level components
        self.brainwave_usecase_factory = lambda: BrainwaveAnalyzeUseCase(
            validator=self.brainwave_validator,
            producer=self.kafka_producer,
            topic_input_raw=self.topic_brainwave_input_raw,
        )

        self.sleep_session_service_factory = lambda: SleepSessionService(
            repo_factory=self.sleep_session_repository_factory,
        )

    # Accessors for FastAPI Depends usage
    def brainwave_usecase(self) -> BrainwaveAnalyzeUseCase:
        return self.brainwave_usecase_factory()

    def session_service(self) -> SleepSessionService:
        return self.sleep_session_service_factory()


# Global container instance
container = Container()
