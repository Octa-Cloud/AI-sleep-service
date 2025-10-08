@echo off
setlocal enabledelayedexpansion

REM Determine script directory and compose file path
set "SCRIPT_DIR=%~dp0"
set "COMPOSE_FILE=%SCRIPT_DIR%docker-compose.yml"

echo Starting MySQL container...
docker compose -f "%COMPOSE_FILE%" up -d mysql
if errorlevel 1 (
  echo Failed to start MySQL via docker compose.
  exit /b 1
)

echo Waiting for MySQL to be healthy...
set /a COUNT=0
set "STATUS=starting"

:wait_loop
for /f "usebackq tokens=*" %%H in (`docker inspect -f "{{.State.Health.Status}}" ai-sleep-mysql-test 2^>NUL`) do set "STATUS=%%H"
if /i "!STATUS!"=="healthy" goto healthy

set /a COUNT+=1
if !COUNT! GEQ 60 goto timeout
rem sleep 2 seconds
timeout /t 2 /nobreak >NUL
goto wait_loop

:healthy
echo MySQL container is up and healthy.
exit /b 0

:timeout
echo Error: MySQL container did not become healthy in time.
exit /b 1


