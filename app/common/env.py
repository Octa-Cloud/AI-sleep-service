from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


@lru_cache(maxsize=None)
def load_service_env() -> bool:
    # Load .env from project root once and cache
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)
    return True


