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
        self.whitelist = ["/health", "/docs", "/openapi.json", "/redoc"]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        
        # Skip authentication for whitelisted paths
        if request.url.path in self.whitelist:
            await self.app(scope, receive, send)
            return

        # nginx auth-url has already validated the request
        # Just extract userNo from the JWT token without validation
        auth_header = request.headers.get(self.token_provider._header_name)
        if auth_header:
            try:
                token = self.token_provider.extract_from_header(auth_header)
                # Decode JWT without verification since nginx already validated it
                import jwt
                claims = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
                user_no = claims.get(self.token_provider._id_claim)
                if user_no:
                    request.state.user_no = int(user_no)
                    request.state.claims = claims
                    await self.app(scope, receive, send)
                    return
            except Exception:
                # If token extraction fails, fall through to error handling
                pass

        # If no valid token found, return 401
        from app.api.domain.application.dto.response.api_response import ApiResponse
        payload = ApiResponse.on_failure("AUTH401", "인증이 필요합니다.").model_dump(mode="json", exclude_none=True)
        response = JSONResponse(status_code=401, content=payload)
        await response(scope, receive, send)
        return


