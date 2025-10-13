from __future__ import annotations

from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Depends, Request

from app.api.domain.application.dto.response.api_response import ApiResponse
from app.api.common.dependencies import Container
from app.api.domain.application.usecase.brainwave.brainwave_analyze_use_case import BrainwaveAnalyzeUseCase
from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService
from app.api.common.exception.custom.session_exceptions import SleepSessionNotFoundApiException


Router = APIRouter(prefix="/api/analysis/brainwave", tags=["brainwave-data"])


def _get_brainwave_usecase(request: Request) -> BrainwaveAnalyzeUseCase:
    container: Container = request.app.container  # type: ignore[attr-defined]
    return container.brainwave_usecase_factory()

def _get_session_service(request: Request) -> SleepSessionService:
    container: Container = request.app.container  # type: ignore[attr-defined]
    return container.session_service()

@Router.patch("")
async def SubmitBrainwave(
    file_instance: Annotated[UploadFile, File(...)],
    use_case: BrainwaveAnalyzeUseCase = Depends(_get_brainwave_usecase),
    session_service: SleepSessionService = Depends(_get_session_service),
    request: Request = None,
):
    edf_bytes = await file_instance.read()
    user_no = int(request.state.user_no)

    current_sleep_session_no = session_service.get_current_sleep_session_no(user_no)
    if current_sleep_session_no is None:
        raise SleepSessionNotFoundApiException()

    await use_case.execute(int(current_sleep_session_no), edf_bytes)
    return ApiResponse.on_success({"accepted": True})

