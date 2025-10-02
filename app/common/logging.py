from __future__ import annotations

import logging
import logging.config
import os
from functools import lru_cache


class ServiceFilter(logging.Filter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        record.service = self._service_name
        record.correlation_id = getattr(record, "correlation_id", "-")
        record.trace_id = getattr(record, "trace_id", "-")
        return True


@lru_cache(maxsize=None)
def _configured_profile() -> str:
    return os.getenv("LOG_PROFILE", "text").lower()


def configure_logging(service_name: str) -> None:
    """Configure structured console logging for multi-service deployment."""

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    profile = _configured_profile()

    formatter: dict[str, object]
    if profile == "json":
        formatter = {
            "format": "%(asctime)s %(levelname)s [%(service)s] %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    else:
        formatter = {
            "format": "%(asctime)s %(levelname)s [%(service)s] %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "service": {
                "()": "app.common.logging.ServiceFilter",
                "service_name": service_name,
            }
        },
        "formatters": {"default": formatter},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "default",
                "filters": ["service"],
            }
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
