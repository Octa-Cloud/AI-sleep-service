from __future__ import annotations

from app.api.common.exception.api_exception import ApiException


class UnauthorizedApiException(ApiException):
    status_code: int = 401
    message: str = "인증이 필요합니다."


class UserNotFoundApiException(ApiException):
    status_code: int = 404
    message: str = "사용자를 찾을 수 없습니다."


