from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.common import config


def compute_sleep_date_from_session_created_at(created_at_utc: datetime) -> datetime.date:
    tz = ZoneInfo(config.REPORT_CUTOFF_TZ)
    if created_at_utc.tzinfo is None:
        created_at_utc = created_at_utc.replace(tzinfo=ZoneInfo("UTC"))
    local = created_at_utc.astimezone(tz)
    cutoff = local.replace(hour=config.REPORT_CUTOFF_HOUR_LOCAL, minute=0, second=0, microsecond=0)
    if local >= cutoff:
        return local.date()
    return (local - timedelta(days=1)).date()


