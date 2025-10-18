from __future__ import annotations

from datetime import datetime

from typing import Annotated
from fastapi import APIRouter, Form, Request, Depends

from app.api.domain.application.dto.response.api_response import ApiResponse
from app.api.common.dependencies import Container
from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService
from app.api.domain.application.usecase.report.sleep_session_finished_use_case import SleepSessionFinishedUseCase
from app.api.common.exception.custom.session_exceptions import SleepSessionNotFoundApiException


Router = APIRouter(prefix="/api/analysis/session", tags=["sleep-session"])

def _get_session_service(request: Request) -> SleepSessionService:
    container: Container = request.app.container 
    return container.session_service()

def _get_session_finished_usecase(request: Request) -> SleepSessionFinishedUseCase:
    container: Container = request.app.container  # type: ignore[attr-defined]
    return container.sleep_session_finished_usecase()

@Router.post("")
async def CreateSleepSession(
    request: Request,
    Service: SleepSessionService = Depends(_get_session_service),
):
    UserNo = int(request.state.user_no)
    Service.begin(UserNo)
    return ApiResponse.on_success()

@Router.delete("")
async def DeleteSleepSession(
    request: Request,
    Service: SleepSessionService = Depends(_get_session_service),
    use_case: SleepSessionFinishedUseCase = Depends(_get_session_finished_usecase),
):
    UserNo = int(request.state.user_no)
    current_sleep_session_no = Service.get_current_sleep_session_no(UserNo)
    if current_sleep_session_no is None:
        raise SleepSessionNotFoundApiException()
    await use_case.execute(int(current_sleep_session_no))
    return ApiResponse.on_success({"finished": True, "session_no": int(current_sleep_session_no)})