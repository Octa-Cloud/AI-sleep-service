from __future__ import annotations

from app.api.common.exception.api_exception import ApiException


class SoundAnalyzeFailApiException(ApiException):
    status_code: int = 500
    message: str = "음성 분석에 실패했습니다."
    code_key: str = "SOUND"


class SoundFormatValidationFailApiException(ApiException):
    status_code: int = 400
    message: str = "지원하지 않는 음성 파일 형식입니다."
    code_key: str = "SOUND_FORMAT"


class SoundChannelValidationFailApiException(ApiException):
    status_code: int = 400
    message: str = "채널 수가 올바르지 않습니다."
    code_key: str = "SOUND_CHANNEL"


class SoundLengthValidationFailApiException(ApiException):
    status_code: int = 400
    message: str = "음성 길이가 허용된 길이를 초과합니다."
    code_key: str = "SOUND_LENGTH"


class SoundSampleRateValidationFailApiException(ApiException):
    status_code: int = 400
    message: str = "샘플링 레이트가 올바르지 않습니다."
    code_key: str = "SOUND_SR"


