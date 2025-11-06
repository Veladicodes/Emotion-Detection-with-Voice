@echo off
REM Quick launcher for Emotion Voice Application
REM Double-click this file to start both backend and frontend

echo.
echo ============================================================
echo   EMOTION VOICE - Starting Application
echo ============================================================
echo.

REM Check if PowerShell is available
where pwsh >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using PowerShell Core...
    pwsh -ExecutionPolicy Bypass -File "%~dp0start_all.ps1"
) else (
    echo Using Windows PowerShell...
    powershell -ExecutionPolicy Bypass -File "%~dp0start_all.ps1"
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo   Error starting application
    echo ============================================================
    echo.
    pause
)

