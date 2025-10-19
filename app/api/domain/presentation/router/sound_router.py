from __future__ import annotations

from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Depends, Request

from app.api.domain.application.dto.response.api_response import ApiResponse
from app.api.common.dependencies import Container
from app.api.common.dependencies.auth import get_user_no
from app.api.domain.application.usecase.sound.sound_analyze_use_case import SoundAnalyzeUseCase
from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService
from app.api.common.exception.custom.session_exceptions import SleepSessionNotFoundApiException

Router = APIRouter(prefix="/api/analysis/sound", tags=["sound-data"])

def _get_sound_usecase(request: Request) -> SoundAnalyzeUseCase:
    container: Container = request.app.container  # type: ignore[attr-defined]
    return container.sound_usecase()

def _get_session_service(request: Request) -> SleepSessionService:
    container: Container = request.app.container  # type: ignore[attr-defined]
    return container.session_service()


@Router.patch("")
async def SubmitSound(
    file_instance: Annotated[UploadFile, File(...)],
    use_case: SoundAnalyzeUseCase = Depends(_get_sound_usecase),
    session_service: SleepSessionService = Depends(_get_session_service),
    user_no: int = Depends(get_user_no),
):
    sound_bytes = await file_instance.read()

    current_sleep_session_no = session_service.get_current_sleep_session_no(user_no)
    if current_sleep_session_no is None:
        raise SleepSessionNotFoundApiException()

    await use_case.execute(int(current_sleep_session_no), sound_bytes)
    return ApiResponse.on_success({"accepted": True})



