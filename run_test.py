#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_root_env() -> None:
    root = Path(__file__).resolve().parent
    env_path = root / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(env_path)
            return
        except Exception:
            pass
        # Minimal fallback parser
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))
        except Exception:
            pass


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tests: app/ (default) or integration_kafka/ (with --e2e)")
    p.add_argument("paths", nargs="*", help="Optional test paths (defaults based on mode)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--e2e", action="store_true", help="Run Kafka E2E integration tests (integration_kafka)")
    p.add_argument("-k", dest="keyword", default=None, help="Pytest -k expression filter")
    p.add_argument("--brokers", default=None, help="Kafka bootstrap servers (default: localhost:29092 for E2E)")
    p.add_argument("--api-base", default=None, help="API base URL for HTTP E2E tests (default: http://localhost:8080)")
    p.add_argument("--", dest="pytest_rest", nargs=argparse.REMAINDER, help="Pass-through args to pytest after --")
    return p.parse_args(argv)


def _run_pytest(args: list[str]) -> int:
    try:
        import pytest  # type: ignore
    except Exception:
        print("pytest is not installed; install with: pip install -r requirements.txt", file=sys.stderr)
        return 2
    return int(pytest.main(args))


def _interactive_args() -> argparse.Namespace:
    print("Select test mode:")
    print("  [1] Standard tests")
    print("  [2] Kafka E2E tests")
    choice = input("Enter choice (1/2): ").strip() or "1"

    # Build a minimal Namespace similar to argparse output
    ns = argparse.Namespace(paths=[], e2e=False, keyword=None, brokers=None, api_base=None, pytest_rest=None)
    if choice == "2":
        ns.e2e = True
        brokers = input("Kafka brokers [default: localhost:29092]: ").strip()
        ns.brokers = brokers or None
        api_base = input("API base URL [default: http://localhost:8080]: ").strip()
        ns.api_base = api_base or None
        print("Ensure Docker services are running: docker compose up -d --build")
    k = input("Optional pytest -k filter (leave empty for none): ").strip()
    ns.keyword = k or None
    return ns


def main(argv: list[str]) -> int:
    _load_root_env()
    ns = _interactive_args() if not argv else _parse_args(argv)

    pytest_args: list[str] = []

    # Force host-accessible endpoints for external services (override .env)
    os.environ["KAFKA_BROKERS"] = "localhost:29092"
    os.environ["API_BASE"] = "http://localhost:8080"

    if ns.e2e:
        # Kafka E2E mode
        os.environ["KAFKA_E2E"] = "1"
        if ns.brokers:
            os.environ["KAFKA_BROKERS"] = ns.brokers
        else:
            # Force host brokers by default (override any .env setting)
            os.environ["KAFKA_BROKERS"] = "localhost:29092"
        if ns.api_base:
            os.environ["API_BASE"] = ns.api_base
        else:
            os.environ.setdefault("API_BASE", "http://localhost:8080")

        print("[info] Running Kafka E2E tests. Ensure Docker services (Kafka + API) are up:")
        print("       docker compose up -d --build")

        # E2E paths
        if ns.paths:
            pytest_args.extend(ns.paths)
        else:
            pytest_args.append("app/tests/integration_kafka")
    else:
        # Standard tests (unit/integration without Kafka)
        os.environ["KAFKA_E2E"] = "0"
        # Default: run only unit tests under app/tests/app
        if ns.paths:
            pytest_args.extend(ns.paths)
        else:
            pytest_args.append("app/tests/app")

    if ns.keyword:
        pytest_args.extend(["-k", ns.keyword])

    if ns.pytest_rest:
        pytest_args.extend(ns.pytest_rest)

    return _run_pytest(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


