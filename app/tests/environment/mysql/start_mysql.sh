#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

export DB_USER="test"
export DB_PASSWORD="testpw"
export DB_HOST="127.0.0.1"
export DB_PORT="3307"
export DB_NAME="sleep_test"

# Start MySQL via docker compose
docker compose -f "$COMPOSE_FILE" up -d mysql

# Wait for readiness
ATTEMPTS=90
SLEEP=2
for ((i=1; i<=ATTEMPTS; i++)); do
  if docker exec ai-sleep-mysql-test mysql -h 127.0.0.1 -uroot -prootpass -e "SELECT 1" >/dev/null 2>&1; then
    echo "MySQL is ready."
    exit 0
  fi
  echo "Waiting for MySQL... ($i/$ATTEMPTS)"
  sleep "$SLEEP"
done

echo "MySQL did not become ready in time." >&2
exit 1
