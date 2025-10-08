from __future__ import annotations

import json
from typing import Any, Callable


def to_bytes(obj: Any, serializer: Callable[[Any], bytes] | None = None) -> bytes:
    if serializer is not None:
        return serializer(obj)
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def from_bytes(data: bytes, deserializer: Callable[[bytes], Any] | None = None) -> Any:
    if deserializer is not None:
        return deserializer(data)
    return json.loads(data.decode("utf-8"))


