#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def main() -> int:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proto_dir = os.path.join(root_dir, "app", "common", "kafka", "dto")
    proto_file = os.path.join(proto_dir, "brainwave.proto")

    try:
        from grpc_tools import protoc  # type: ignore
    except Exception:
        print("grpcio-tools is not installed. Install with: pip install grpcio-tools", file=sys.stderr)
        return 1

    args = [
        "protoc",
        f"-I{proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        proto_file,
    ]
    code = protoc.main(args)
    if code != 0:
        print(f"protoc failed with exit code {code}", file=sys.stderr)
        return int(code)
    print(f"Generated protobufs under {proto_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


