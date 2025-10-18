from __future__ import annotations

from datetime import datetime

from typing import Annotated
from fastapi import APIRouter, Form, Request, Depends

from app.api.domain.application.dto.response.api_response import ApiResponse
from app.api.common.dependencies import Container
from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService


Router = APIRouter(prefix="/api/analysis/session", tags=["sleep-session"])

def _get_session_service(request: Request) -> SleepSessionService:
    container: Container = request.app.container 
    return container.session_service()

@Router.post("")
async def CreateSleepSession(
    request: Request,
    Service: SleepSessionService = Depends(_get_session_service),
):
    print(f"CreateSleepSession: request.state = {request.state.__dict__}")
    if not hasattr(request.state, 'user_no'):
        print("CreateSleepSession: user_no not found in request.state")
        return ApiResponse.on_error("Authentication required")
    UserNo = int(request.state.user_no)
    Service.begin(UserNo)
    return ApiResponse.on_success()

@Router.delete("")
async def DeleteSleepSession(
    request: Request,
    Service: SleepSessionService = Depends(_get_session_service),
):
    UserNo = int(request.state.user_no)
    Service.finish(UserNo)
    return ApiResponse.on_success()