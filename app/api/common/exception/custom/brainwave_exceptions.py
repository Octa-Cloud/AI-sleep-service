from __future__ import annotations

from app.api.common.exception.api_exception import ApiException
from app.api.common.exception.custom.analyze_fail_exception import AnalyzeFailException


class BrainwaveAnalyzeFailApiException(AnalyzeFailException):
    status_code: int = 500
    message: str = "뇌파 분석에 실패했습니다."


class BrainwaveFormatValidationFailApiException(ApiException):
    status_code: int = 400
    message: str = "올바르지 않은 파일 형태입니다."


class BrainwaveChannelValidationFailApiException(ApiException):
    status_code: int = 400
    message: str = "올바르지 않은 채널입니다."


class BrainwaveLengthValidationFailApiException(ApiException):
    status_code: int = 400
    message: str = "데이터의 길이가 청크 사이즈를 초과합니다."


