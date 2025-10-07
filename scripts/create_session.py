#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional


def _make_token(user_no: int) -> Optional[str]:
    try:
        import jwt  # type: ignore
    except Exception:
        return None

    secret = os.getenv("JWT_SECRET")
    alg = os.getenv("JWT_ALGORITHM", "HS256")
    sub = os.getenv("JWT_ACCESS_SUBJECT", "AccessToken")
    id_key = os.getenv("JWT_ID_CLAIM", "id")
    if not secret:
        return None
    if secret and secret.startswith("\"") and secret.endswith("\""):
        secret = secret.strip("\"")
    payload = {"sub": sub, id_key: int(user_no), "exp": int(time.time()) + 3600}
    return jwt.encode(payload, secret, algorithm=alg)


def create_session(url: str, token: Optional[str]) -> int:
    try:
        import requests  # type: ignore
    except Exception:
        print("Please install requests: python -m pip install requests", file=sys.stderr)
        return 2

    headers = {}
    if token:
        headers[os.getenv("JWT_TOKEN_HEADER", "Authorization")] = f"{os.getenv('JWT_BEARER_PREFIX','Bearer')} {token}"

    resp = requests.post(url, headers=headers, timeout=30)
    print(f"status={resp.status_code}\nbody={resp.text}")
    return 0 if resp.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a sleep session via API")
    parser.add_argument("--url", default="http://localhost:8080/api/sleep/session/begin", help="Session create endpoint URL")
    parser.add_argument("--token", dest="token", default=None, help="Bearer token. If omitted, tries to mint from JWT_* envs")
    parser.add_argument("--user-no", dest="user_no", type=int, default=1, help="User no for token minting if --token not provided")
    parser.add_argument("--env-file", dest="env_file", default=None, help="Optional path to .env to load before minting")
    args = parser.parse_args()

    if args.env_file:
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(args.env_file)
        except Exception:
            print("Warning: python-dotenv not installed; --env-file ignored", file=sys.stderr)

    token = args.token or _make_token(args.user_no)
    if not token:
        print("Failed to get token. Ensure JWT_* env vars are set or pass --token.", file=sys.stderr)
        return 2

    return create_session(args.url, token)


if __name__ == "__main__":
    raise SystemExit(main())


