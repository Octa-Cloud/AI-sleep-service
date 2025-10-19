from __future__ import annotations

from typing import Annotated
from fastapi import Depends, Header, HTTPException
from app.api.common.security.token_provider import TokenProvider

def get_user_no(authorization: Annotated[str, Header()]) -> int:
    """
    Authorization 헤더에서 토큰을 추출하고 userNo를 반환합니다.
    """
    try:
        token_provider = TokenProvider()
        token = token_provider.extract_from_header(authorization)
        user_no = token_provider.get_user_no(token)
        return user_no
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
