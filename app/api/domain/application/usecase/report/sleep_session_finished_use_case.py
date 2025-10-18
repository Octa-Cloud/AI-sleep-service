from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
import logging
from sqlalchemy.exc import IntegrityError

from app.common import config
from app.common.time_utils import compute_sleep_date_from_session_created_at
from app.api.domain.application.service.brainwave.sleep_level_service import SleepLevelService
from app.api.domain.application.service.daily_report.daily_report_service import DailyReportService
from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService
from app.api.common.tsid import generate_int as generate_tsid_int
from app.api.common.decorator.session_scope import session_scope


class SleepSessionFinishedUseCase:
    def __init__(
        self,
        sleep_level_service: SleepLevelService,
        daily_report_service: DailyReportService,
        sleep_session_service: SleepSessionService,
        session_repo_factory,
        producer_factory,
    ) -> None:
        self._levels = sleep_level_service
        self._daily = daily_report_service
        self._sessions = sleep_session_service
        self._session_repo_factory = session_repo_factory
        self._producer_factory = producer_factory  # returns aiokafka producer or wrapper
        self._logger = logging.getLogger("report.usecase")

    @session_scope
    async def execute(self, session_no: int, session=None) -> None:
        # 0) load session then finish it to stamp finished_at
        session_repo = self._session_repo_factory(session=session)
        sess = session_repo.find_by_id(int(session_no))
        self._sessions.finish(int(sess.user_no))
        user_no = int(sess.user_no)
        created_at: datetime = sess.created_at if sess.created_at.tzinfo else sess.created_at.replace(tzinfo=timezone.utc)
        sleep_date = compute_sleep_date_from_session_created_at(created_at)

        # 1.5) Insert placeholder row (INSERT-only). On failure, abort without publishing.
        try:
            self._daily.insert_placeholder(int(session_no), user_no, created_at, session=session)
        except IntegrityError:
            # Duplicate PK (already inserted). Log clearly without stacktrace and abort.
            self._logger.error(
                "daily_placeholder_duplicate_skip",
                extra={"session_no": int(session_no), "user_no": user_no},
            )
            return
        except Exception as e:
            # Any other failure - log and abort without stacktrace
            self._logger.error(
                "daily_placeholder_insert_failed",
                extra={"session_no": int(session_no), "user_no": user_no, "err": str(e)},
            )
            return

        # 2) gather levels
        level_vos = self._levels.get_levels_by_session(int(session_no), session=session)

        # 3) optional existing daily report by date
        existing = self._daily.get_by_date(user_no, sleep_date, session=session)

        # 4) build and publish protobufs
        from app.common.kafka.dto import report_pb2 as rp  # type: ignore
        prod = self._producer_factory()
        trace_id = str(generate_tsid_int())

        # DailyReportInput
        dri = rp.DailyReportInput(
            session_no=int(session_no),
            user_no=user_no,
            created_at_ms=int(created_at.timestamp() * 1000),
            sleep_date=str(sleep_date),
        )
        for v in level_vos:
            dri.levels.add(level=int(v.level), recorded_at_ms=int(v.recorded_at.timestamp() * 1000))
        if existing is not None:
            dri.daily_report.score = int(existing.score or 0)
            if existing.total_sleep_minutes is not None:
                dri.daily_report.total_sleep_minutes = int(existing.total_sleep_minutes)
            if existing.deep_sleep_ratio is not None:
                dri.daily_report.deep_ratio = float(existing.deep_sleep_ratio)
            if existing.light_sleep_ratio is not None:
                dri.daily_report.light_ratio = float(existing.light_sleep_ratio)
            if existing.rem_sleep_ratio is not None:
                dri.daily_report.rem_ratio = float(existing.rem_sleep_ratio)

        await prod.send_and_wait(
            config.TOPIC_DAILY_REPORT_INPUT,
            key=f"{int(session_no)}:{trace_id}".encode(),
            value=dri.SerializeToString(),
        )

        # PeriodicReportInput - weekly & monthly
        for dtype in (rp.WEEKLY, rp.MONTHLY):
            pri = rp.PeriodicReportInput(
                session_no=int(session_no),
                user_no=user_no,
                sleep_date=str(sleep_date),
                duration_type=dtype,
            )
            wmtag = "W" if int(dtype) == int(rp.WEEKLY) else "M"
            await prod.send_and_wait(
                config.TOPIC_PERIODIC_REPORT_INPUT,
                key=f"{int(session_no)}:{trace_id}:{wmtag}".encode(),
                value=pri.SerializeToString(),
            )


