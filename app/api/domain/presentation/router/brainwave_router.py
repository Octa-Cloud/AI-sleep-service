from __future__ import annotations

from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Form, Depends, Request

from app.api.domain.application.dto.response.api_response import ApiResponse
from app.api.common.dependencies import Container
from app.api.domain.application.usecase.brainwave.brainwave_analyze_use_case import BrainwaveAnalyzeUseCase


Router = APIRouter(prefix="/api/sleep/data/brainwave", tags=["brainwave-data"])


def _get_brainwave_usecase(request: Request) -> BrainwaveAnalyzeUseCase:
    container: Container = request.app.container  # type: ignore[attr-defined]
    return container.brainwave_usecase_factory()

@Router.patch("/")
async def SubmitBrainwave(
    file_instance: Annotated[UploadFile, File(...)],
    sleep_session_no: Annotated[int, Form(...)],
    use_case: BrainwaveAnalyzeUseCase = Depends(_get_brainwave_usecase),
):
    edf_bytes = await file_instance.read()
    await use_case.execute(int(sleep_session_no), edf_bytes)
    return ApiResponse.on_success({"accepted": True})

