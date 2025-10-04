#!/usr/bin/env python3
from __future__ import annotations

import os

from app.common.env import load_service_env


def main() -> None:
    load_service_env()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))

    import uvicorn

    uvicorn.run("app.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()



