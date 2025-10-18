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
        print(f"AuthMiddleware: __call__ started, scope type: {scope.get('type')}")
        
        if scope["type"] != "http":
            print(f"AuthMiddleware: Non-HTTP request, passing through")
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        
        print(f"AuthMiddleware: Processing HTTP request to {request.url.path}")
        print(f"AuthMiddleware: Request method: {request.method}")
        print(f"AuthMiddleware: Request headers: {dict(request.headers)}")
        
        # Skip authentication for whitelisted paths
        if request.url.path in self.whitelist:
            print(f"AuthMiddleware: Skipping authentication for whitelisted path: {request.url.path}")
            await self.app(scope, receive, send)
            return

        # Direct JWT token validation (nginx auth-url disabled)
        print(f"AuthMiddleware: Starting JWT token validation")
        auth_header = request.headers.get("Authorization")
        print(f"AuthMiddleware: Authorization header: {auth_header}")
        
        if auth_header:
            try:
                print(f"AuthMiddleware: Extracting token from header")
                token = self.token_provider.extract_from_header(auth_header)
                print(f"AuthMiddleware: Extracted token: {token[:50]}...")
                
                print(f"AuthMiddleware: Verifying JWT token")
                # Verify JWT token with signature and expiration
                claims = self.token_provider.verify_and_decode(token)
                print(f"AuthMiddleware: Token claims: {claims}")
                
                user_no = claims.get("id")
                print(f"AuthMiddleware: User ID from claims: {user_no}")
                
                if user_no:
                    request.state.user_no = int(user_no)
                    request.state.claims = claims
                    print(f"AuthMiddleware: Successfully authenticated user {user_no}, proceeding to app")
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
                import traceback
                print(f"AuthMiddleware error traceback: {traceback.format_exc()}")
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


