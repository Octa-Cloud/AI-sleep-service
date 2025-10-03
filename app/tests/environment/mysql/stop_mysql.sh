#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Stop and remove containers and anonymous volumes
docker compose -f "$COMPOSE_FILE" down --volumes
