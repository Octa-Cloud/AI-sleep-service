from datetime import datetime
from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, Field

T = TypeVar("T")
class ApiResponse(BaseModel, Generic[T]):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    code: str
    message: str
    result: Optional[T] = None

    model_config = {"exclude_none": True}

    @staticmethod
    def on_success(result: Optional[T] = None) -> "ApiResponse[T]":
        return ApiResponse(code="COMMON200", message="요청에 성공하였습니다", result=result)

    @staticmethod
    def on_failure(code: str, message: str, result: Optional[T] = None) -> "ApiResponse[T]":
        return ApiResponse(code=code, message=message, result=result)