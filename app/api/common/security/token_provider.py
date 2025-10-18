from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from app.api.common.exception.custom.auth_exceptions import UnauthorizedApiException

logger = logging.getLogger(__name__)


class TokenProvider:

    def __init__(self, secret: Optional[str] = None, algorithm: Optional[str] = None) -> None:
        # Ensure .env is loaded before reading
        self._secret: str = secret or os.getenv("JWT_SECRET", "")
        self._algorithm: str = algorithm or os.getenv("JWT_ALGORITHM", "HS256")

        # Env-driven token parsing/config
        self._access_subject: str = os.getenv("JWT_ACCESS_SUBJECT", "AccessToken")
        self._refresh_subject: str = os.getenv("JWT_REFRESH_SUBJECT", "RefreshToken")
        self._header_name: str = os.getenv("JWT_TOKEN_HEADER", "Authorization")
        self._bearer_prefix: str = os.getenv("JWT_BEARER_PREFIX", "Bearer")
        self._id_claim: str = os.getenv("JWT_ID_CLAIM", "id")

        if not self._secret:
            # Treat missing secret as a server-side misconfiguration surfaced as 401 to callers of this component.
            raise UnauthorizedApiException()

    def verify_and_decode(self, token: str) -> Dict[str, Any]:
        if not token:
            raise UnauthorizedApiException()

        try:
            claims: Dict[str, Any] = jwt.decode(
                token,
                key=self._secret,
                algorithms=self._algorithm,
                options={"require": ["exp"], "verify_exp": True},
            )
            return claims
        except ExpiredSignatureError as e:
            exc = UnauthorizedApiException()
            exc.message = "토큰이 만료되었습니다."
            raise exc
        except InvalidTokenError as e:
            exc = UnauthorizedApiException()
            exc.message = "유효하지 않은 토큰입니다."
            raise exc
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise UnauthorizedApiException()

    def get_claims(self, token: str) -> Dict[str, Any]:
        return self.verify_and_decode(token)

    def get_user_no(self, token: str) -> int:
        claims = self.verify_and_decode(token)
        try:
            return int(claims[self._id_claim])
        except Exception:
            exc = UnauthorizedApiException()
            exc.message = "토큰에 사용자 정보가 없습니다."
            raise exc

    # Helpers using env-driven config
    def extract_from_header(self, header_value: Optional[str]) -> str:
        if not header_value:
            raise UnauthorizedApiException()
        prefix = f"{self._bearer_prefix} "
        if not header_value.startswith(prefix):
            exc = UnauthorizedApiException()
            exc.message = "인증 헤더 형식이 올바르지 않습니다."
            raise exc
        token = header_value[len(prefix) :].strip()
        if not token:
            raise UnauthorizedApiException()
        return token

    def validate_access_subject(self, claims: Dict[str, Any]) -> None:
        if claims.get("sub") != self._access_subject:
            exc = UnauthorizedApiException()
            exc.message = "유효하지 않은 토큰 타입입니다."
            raise exc

    def validate_refresh_subject(self, claims: Dict[str, Any]) -> None:
        if claims.get("sub") != self._refresh_subject:
            exc = UnauthorizedApiException()
            exc.message = "유효하지 않은 토큰 타입입니다."
            raise exc


