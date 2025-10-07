#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional


def _strip_quotes(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("\"") and value.endswith("\""):
        return value.strip("\"")
    return value


def mint_token(user_no: int, exp_seconds: int, subject: Optional[str] = None) -> Optional[str]:
    try:
        import jwt  # type: ignore
    except Exception:
        print("Please install pyjwt: python -m pip install PyJWT", file=sys.stderr)
        return None

    secret = _strip_quotes(os.getenv("JWT_SECRET"))
    if not secret:
        print("JWT_SECRET is not set", file=sys.stderr)
        return None

    alg = os.getenv("JWT_ALGORITHM", "HS256")
    sub = subject or os.getenv("JWT_ACCESS_SUBJECT", "AccessToken")
    id_key = os.getenv("JWT_ID_CLAIM", "id")

    payload = {
        "sub": sub,
        id_key: int(user_no),
        "exp": int(time.time()) + int(exp_seconds),
    }
    return jwt.encode(payload, secret, algorithm=alg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a signed JWT for testing")
    parser.add_argument("--user-no", type=int, default=1, help="User number to embed in the token")
    parser.add_argument("--exp-seconds", type=int, default=3600, help="Token TTL in seconds")
    parser.add_argument("--subject", default=None, help="Override JWT subject (sub)")
    parser.add_argument("--env-file", default=None, help="Optional path to .env to load before minting")
    parser.add_argument("--as-header", action="store_true", help="Print as Authorization header")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Print JSON with token and claims info")
    args = parser.parse_args()

    if args.env_file:
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(args.env_file)
        except Exception:
            print("Warning: python-dotenv not installed; --env-file ignored", file=sys.stderr)

    token = mint_token(args.user_no, args.exp_seconds, args.subject)
    if not token:
        return 1

    if args.as_json:
        id_key = os.getenv("JWT_ID_CLAIM", "id")
        out = {
            "token": token,
            "header_name": os.getenv("JWT_TOKEN_HEADER", "Authorization"),
            "bearer_prefix": os.getenv("JWT_BEARER_PREFIX", "Bearer"),
            "user_no": int(args.user_no),
            "id_claim": id_key,
        }
        print(json.dumps(out, ensure_ascii=False))
        return 0

    if args.as_header:
        header_name = os.getenv("JWT_TOKEN_HEADER", "Authorization")
        bearer = os.getenv("JWT_BEARER_PREFIX", "Bearer")
        print(f"{header_name}: {bearer} {token}")
        return 0

    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


