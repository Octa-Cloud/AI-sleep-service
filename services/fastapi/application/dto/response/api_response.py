from datetime import datetime
from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T")
class ApiResponse(Generic[T], BaseModel):
    timestamp: datetime
    code: str
    message: str
    result: Optional[T] = None

    model_config = { "exclude_none": True }

    def __init__(self, code, message, result):
        self.code = code
        self.message = message
        self.result = result
        self.timestamp = datetime.now()

    def on_success(result = None):
        return ApiResponse('COMMON200', '요청에 성공하였습니다', result)

    def on_failure(code, message, result = None):
        return ApiResponse(code, message, result)