from __future__ import annotations

from datetime import datetime

from typing import Annotated
from fastapi import APIRouter, Form, Request, Depends

from app.api.domain.application.dto.response.api_response import ApiResponse
from app.api.common.dependencies import Container
from app.api.common.dependencies.auth import get_user_no
from app.api.domain.application.service.sleep_session.sleep_session_service import SleepSessionService


Router = APIRouter(prefix="/api/analysis/session", tags=["sleep-session"])

def _get_session_service(request: Request) -> SleepSessionService:
    print(f"_get_session_service: Function called")
    try:
        container: Container = request.app.container 
        print(f"_get_session_service: Got container: {container}")
        service = container.session_service()
        print(f"_get_session_service: Got session service: {service}")
        return service
    except Exception as e:
        print(f"_get_session_service: Exception occurred: {e}")
        import traceback
        print(f"_get_session_service: Exception traceback: {traceback.format_exc()}")
        raise

@Router.post("")
async def CreateSleepSession(
    request: Request,
    Service: SleepSessionService = Depends(_get_session_service),
    UserNo: int = Depends(get_user_no),
):
    print(f"CreateSleepSession: Function called")
    print(f"CreateSleepSession: request.state = {request.state.__dict__}")
    print(f"CreateSleepSession: request.headers = {dict(request.headers)}")
    print(f"CreateSleepSession: request.url = {request.url}")
    print(f"CreateSleepSession: Extracted UserNo = {UserNo}")
    
    try:
        print(f"CreateSleepSession: Calling Service.begin({UserNo})")
        Service.begin(UserNo)
        print(f"CreateSleepSession: Service.begin completed successfully")
        
        result = ApiResponse.on_success()
        print(f"CreateSleepSession: Returning success response: {result}")
        return result
        
    except Exception as e:
        print(f"CreateSleepSession: Exception occurred: {e}")
        import traceback
        print(f"CreateSleepSession: Exception traceback: {traceback.format_exc()}")
        return ApiResponse.on_failure("COMMON500", f"Internal server error: {str(e)}")

@Router.delete("")
async def DeleteSleepSession(
    request: Request,
    Service: SleepSessionService = Depends(_get_session_service),
    UserNo: int = Depends(get_user_no),
):
    Service.finish(UserNo)
    return ApiResponse.on_success()