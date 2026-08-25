@echo off
setlocal enabledelayedexpansion
title NNLC Auto Train

set PYTHON=%~dp0.venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo [ERROR] Python venv not found
    echo Please run: cd /d "%~dp0" ^&^& python -m venv .venv ^&^& .venv\Scripts\pip install -e .
    pause
    exit /b 1
)

if not exist "%~dp0nnlc_auto_train.py" (
    echo [ERROR] Training script not found
    pause
    exit /b 1
)

if "%~1"=="" (
    echo.
    echo ========================================
    echo   NNLC Auto Train - Interactive Mode
    echo ========================================
    echo.
    cd /d "%~dp0"
    "%PYTHON%" "%~dp0nnlc_auto_train.py"
) else (
    echo.
    echo ========================================
    echo   NNLC Auto Train - Quick Mode
    echo ========================================
    echo.
    echo Data dir: %~1
    echo.
    set /p CAR_NAME=Car name e.g. BYD_TANG_DMI_24:
    if "!CAR_NAME!"=="" (
        echo [ERROR] Car name cannot be empty
        pause
        exit /b 1
    )
    cd /d "%~dp0"
    "%PYTHON%" "%~dp0nnlc_auto_train.py" --data "%~1" --car !CAR_NAME!
)

echo.
echo Done.
pause