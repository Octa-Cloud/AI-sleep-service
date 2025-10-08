from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    _load_dotenv = None


@lru_cache(maxsize=None)
def load_service_env() -> bool:
    # Load .env from project root once and cache
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        if _load_dotenv is not None:
            _load_dotenv(env_path, override=False)  # type: ignore[misc]
        else:
            # Minimal .env parser fallback (KEY=VALUE, ignore comments/blank lines)
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            except Exception:
                pass
    return True


