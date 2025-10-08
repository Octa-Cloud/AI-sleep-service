from __future__ import annotations

from app.api.common.exception.api_exception import ApiException


class SleepSessionExistsApiException(ApiException):
    status_code: int = 409
    message: str = "세션이 이미 존재합니다."
    code_key: str = "SESSION_EXISTS"


class SleepSessionNotFoundApiException(ApiException):
    status_code: int = 404
    message: str = "진행 중인 수면 세션이 없습니다."
    code_key: str = "SESSION_NOT_FOUND"


