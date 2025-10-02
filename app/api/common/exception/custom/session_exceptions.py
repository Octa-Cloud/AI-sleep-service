from __future__ import annotations

from app.api.common.exception.api_exception import ApiException


class SleepSessionExistsApiException(ApiException):
    status_code: int = 409
    message: str = "세션이 이미 존재합니다."


