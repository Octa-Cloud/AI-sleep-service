from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.domain.application.dto.response.api_response import ApiResponse

from .api_exception import ApiException
from .common_exceptions import InternalApiException


logger = logging.getLogger(__name__)


def _to_response_payload(code: str, message: str, result: Any = None) -> dict[str, Any]:
    # Ensure datetimes are encoded and None fields are excluded
    return ApiResponse.on_failure(code, message, result).model_dump(mode="json", exclude_none=True)


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(ApiException)
    async def handle_api_exception(request: Request, exc: ApiException) -> JSONResponse:
        logger.warning("API exception", exc_info=exc)
        payload = _to_response_payload(f"API{exc.status_code}", exc.message)
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.debug("Request validation error: %s", exc.errors())
        payload = _to_response_payload("COMMON400", "요청 형식이 올바르지 않습니다.", exc.errors())
        return JSONResponse(status_code=400, content=payload)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        logger.info("HTTP exception: %s", exc.detail)
        payload = _to_response_payload(f"HTTP{exc.status_code}", str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - defensive
        logger.exception("Unhandled server error")
        fallback = InternalApiException()
        payload = _to_response_payload(f"API{fallback.status_code}", fallback.message)
        return JSONResponse(status_code=fallback.status_code, content=payload)
