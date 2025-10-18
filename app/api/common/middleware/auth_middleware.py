from __future__ import annotations

import jwt
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
        
        print(f"AuthMiddleware: Processing request to {request.url.path}")
        
        # Skip authentication for whitelisted paths
        if request.url.path in self.whitelist:
            print(f"AuthMiddleware: Skipping authentication for whitelisted path: {request.url.path}")
            await self.app(scope, receive, send)
            return

        # Direct JWT token validation (nginx auth-url disabled)
        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                token = self.token_provider.extract_from_header(auth_header)
                # Verify JWT token with signature and expiration
                claims = self.token_provider.verify_and_decode(token)
                user_no = claims.get("id")
                if user_no:
                    request.state.user_no = int(user_no)
                    request.state.claims = claims
                    await self.app(scope, receive, send)
                    return
                else:
                    print(f"AuthMiddleware error: No user_no in claims: {claims}")
                    return JSONResponse(
                        status_code=401,
                        content={"message": "Invalid token: missing user ID"}
                    )
            except Exception as e:
                # Log the error for debugging
                print(f"AuthMiddleware error: {e}")
                return JSONResponse(
                    status_code=401,
                    content={"message": "Invalid token"}
                )
        
        # No Authorization header found
        print(f"AuthMiddleware error: No Authorization header found")
        return JSONResponse(
            status_code=401,
            content={"message": "Missing Authorization header"}
        )


