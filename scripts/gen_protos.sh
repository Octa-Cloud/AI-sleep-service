#!/usr/bin/env bash
set -euo pipefail

# Prefer python3; fallback to python
PY_BIN=${PY_BIN:-python3}
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN=python
fi

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PROTO_DIR="$ROOT_DIR/app/common/kafka/dto"

"$PY_BIN" -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$PROTO_DIR" \
  --grpc_python_out="$PROTO_DIR" \
  "$PROTO_DIR/brainwave.proto"

echo "Generated protobufs under $PROTO_DIR"


