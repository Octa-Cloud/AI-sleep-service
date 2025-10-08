@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "COMPOSE_FILE=%SCRIPT_DIR%docker-compose.yml"

echo Stopping and removing MySQL container and volumes...
docker compose -f "%COMPOSE_FILE%" down --volumes
if errorlevel 1 (
  echo Error: Failed to stop MySQL container.
  exit /b 1
)

echo MySQL container stopped and removed successfully.
exit /b 0


