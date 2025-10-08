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
    # handle quoted secrets like "..."
    if secret and secret.startswith("\"") and secret.endswith("\""):
        secret = secret.strip("\"")
    payload = {"sub": sub, id_key: int(user_no), "exp": int(time.time()) + 3600}
    return jwt.encode(payload, secret, algorithm=alg)


def send_request(url: str, edf_path: Path, session_no: int, token: Optional[str]) -> int:
    try:
        import requests  # type: ignore
    except Exception:
        print("Please install requests: python -m pip install requests", file=sys.stderr)
        return 2

    headers = {}
    if token:
        headers[os.getenv("JWT_TOKEN_HEADER", "Authorization")] = f"{os.getenv('JWT_BEARER_PREFIX','Bearer')} {token}"

    with edf_path.open("rb") as f:
        files = {"file_instance": (edf_path.name, f, "application/octet-stream")}
        data = {"sleep_session_no": str(int(session_no))}
        resp = requests.patch(url.rstrip("/") + "/", headers=headers, files=files, data=data, timeout=60)
    print(f"status={resp.status_code}\nbody={resp.text}")
    return 0 if resp.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Send brainwave EDF to API (multipart/form-data)")
    parser.add_argument("--url", default="http://localhost:8080/api/sleep/data/brainwave/", help="Endpoint URL")
    parser.add_argument("--file", dest="file", help="Path to EDF file")
    parser.add_argument("--session", dest="session", type=int, default=1, help="Sleep session no")
    parser.add_argument("--token", dest="token", default=None, help="Bearer token. If omitted, tries to mint from JWT_* envs")
    parser.add_argument("--user-no", dest="user_no", type=int, default=123, help="User no for token minting if --token not provided")
    parser.add_argument("--mint-token", dest="mint", action="store_true", help="Print a minted token and exit")
    parser.add_argument("--env-file", dest="env_file", default=None, help="Optional path to .env to load before minting")
    args = parser.parse_args()

    # Optionally load .env first
    if args.env_file:
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(args.env_file)
        except Exception:
            print("Warning: python-dotenv not installed; --env-file ignored", file=sys.stderr)

    # Mint-only mode
    if args.mint:
        tok = args.token or _make_token(args.user_no)
        if not tok:
            print("Failed to mint token. Ensure JWT_* env vars are set.", file=sys.stderr)
            return 1
        print(tok)
        return 0

    if not args.file:
        print("--file is required unless --mint-token is used", file=sys.stderr)
        return 2

    edf_path = Path(args.file)
    if not edf_path.exists():
        print(f"EDF not found: {edf_path}", file=sys.stderr)
        return 2

    token = args.token or _make_token(args.user_no)
    return send_request(args.url, edf_path, args.session, token)


if __name__ == "__main__":
    raise SystemExit(main())


