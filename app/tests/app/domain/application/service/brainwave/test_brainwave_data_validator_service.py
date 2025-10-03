from __future__ import annotations
# 이 파일은 뇌파 데이터 유효성 검사 로직이 잘못된 입력을 거부하는지 테스트합니다.

import pytest

from app.api.domain.application.service.brainwave.brainwave_data_validator_service import BrainwaveDataValidatorService
from app.api.common.exception.custom.brainwave_exceptions import (
    BrainwaveFormatValidationFailApiException,
)


def test_validate_rejects_empty_bytes():
    svc = BrainwaveDataValidatorService()
    with pytest.raises(BrainwaveFormatValidationFailApiException):
        svc.validate(b"")


