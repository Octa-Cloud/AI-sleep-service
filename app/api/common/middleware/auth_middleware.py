from __future__ import annotations

from typing import Callable, List

from fastapi import Request
from fastapi.responses import JSONResponse

from app.api.common.exception.custom.auth_exceptions import UnauthorizedApiException
from app.api.common.security.token_provider import TokenProvider


class AuthMiddleware:
    def __init__(self, app, token_provider: TokenProvider | None = None) -> None:
        self.app = app
        self.token_provider = token_provider or TokenProvider()
        self.whitelist = []

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)

        auth_header = request.headers.get(self.token_provider._header_name)
        try:
            token = self.token_provider.extract_from_header(auth_header)
            claims = self.token_provider.verify_and_decode(token)
            self.token_provider.validate_access_subject(claims)
            request.state.user_no = self.token_provider.get_user_no(token)
            request.state.claims = claims

        except UnauthorizedApiException as exc:
            from app.api.domain.application.dto.response.api_response import ApiResponse
            code = exc.build_code()
            payload = ApiResponse.on_failure(code, exc.message).model_dump(mode="json", exclude_none=True)
            response = JSONResponse(status_code=exc.status_code, content=payload)
            await response(scope, receive, send)
            return

        except Exception:
            from app.api.domain.application.dto.response.api_response import ApiResponse
            payload = ApiResponse.on_failure("AUTH401", "인증이 필요합니다.").model_dump(mode="json", exclude_none=True)
            response = JSONResponse(status_code=401, content=payload)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


