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
from app.api.domain.application.usecase.report.sleep_session_finished_use_case import SleepSessionFinishedUseCase
from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService
from app.api.domain.application.service.sound.sound_data_validator_service import SoundDataValidatorService
from app.api.domain.application.usecase.sound.sound_analyze_use_case import SoundAnalyzeUseCase
from app.api.domain.application.service.sound.sound_event_service import SoundEventService
from app.api.domain.infra.repository.sound_event_repository_impl import SqlAlchemySoundEventRepository
from app.api.domain.application.service.daily_report.daily_report_service import DailyReportService
from app.api.domain.infra.repository.daily_report_repository_impl import SqlAlchemyDailyReportRepository
from app.api.domain.application.service.periodic_report.periodic_report_service import PeriodicReportService
from app.api.domain.infra.repository.periodic_report_repository_impl import SqlAlchemyPeriodicReportRepository


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

        # Report repository factories
        self.daily_report_repository_factory: Callable[..., SqlAlchemyDailyReportRepository] = (
            lambda *, session: SqlAlchemyDailyReportRepository(session=session)
        )
        self.periodic_report_repository_factory: Callable[..., SqlAlchemyPeriodicReportRepository] = (
            lambda *, session: SqlAlchemyPeriodicReportRepository(session=session)
        )

        # Report services
        self.daily_report_service_factory = lambda: DailyReportService(
            repo_factory=self.daily_report_repository_factory
        )
        self.periodic_report_service_factory = lambda: PeriodicReportService(
            repo_factory=self.periodic_report_repository_factory
        )

        # Kafka producer client (to be implemented in app/common/kafka)
        try:
            from app.common.kafka.producer import KafkaProducerClient  # type: ignore
            kafka_bootstrap = os.getenv("KAFKA_BROKERS", "localhost:9092")
            self.kafka_producer = KafkaProducerClient(bootstrap_servers=kafka_bootstrap)
        except Exception:
            self.kafka_producer = None  # type: ignore
        self.topic_brainwave_input_raw = os.getenv("TOPIC_BRAINWAVE_INPUT_RAW", "brainwave.input.raw")
        self.topic_sound_input_raw = os.getenv("TOPIC_SOUND_INPUT_RAW", "sound.input.raw")

        # Sound services
        self.sound_validator = SoundDataValidatorService()
        self.sound_event_service = SoundEventService(
            repo_factory=lambda **kw: SqlAlchemySoundEventRepository(session=kw.get("session", self.session_factory()))
        )

        # Factories for higher-level components
        self.brainwave_usecase_factory = lambda: BrainwaveAnalyzeUseCase(
            validator=self.brainwave_validator,
            producer=self.kafka_producer,
            topic_input_raw=self.topic_brainwave_input_raw,
        )

        self.sound_usecase_factory = lambda: SoundAnalyzeUseCase(
            validator=self.sound_validator,
            producer=self.kafka_producer,
            topic_input_raw=self.topic_sound_input_raw,
        )

        # Report: SleepSessionFinishedUseCase factory
        def _producer_factory():
            # return underlying aiokafka producer for send_and_wait
            return getattr(self.kafka_producer, "_producer", None)

        self.sleep_session_finished_usecase_factory = lambda: SleepSessionFinishedUseCase(
            sleep_level_service=self.brainwave_sleeplevel,
            daily_report_service=self.daily_report_service_factory(),
            sleep_session_service=self.sleep_session_service_factory(),
            session_repo_factory=self.sleep_session_repository_factory,
            producer_factory=_producer_factory,
        )

        self.sleep_session_service_factory = lambda: SleepSessionService(
            repo_factory=self.sleep_session_repository_factory,
        )

    # Accessors for FastAPI Depends usage
    def brainwave_usecase(self) -> BrainwaveAnalyzeUseCase:
        return self.brainwave_usecase_factory()

    def session_service(self) -> SleepSessionService:
        return self.sleep_session_service_factory()

    def sound_usecase(self) -> SoundAnalyzeUseCase:
        return self.sound_usecase_factory()
    def sleep_session_finished_usecase(self) -> SleepSessionFinishedUseCase:
        return self.sleep_session_finished_usecase_factory()

    def daily_report_service(self) -> DailyReportService:
        return self.daily_report_service_factory()

    def periodic_report_service(self) -> PeriodicReportService:
        return self.periodic_report_service_factory()


# Global container instance
container = Container()
