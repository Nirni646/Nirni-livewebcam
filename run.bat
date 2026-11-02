@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [Nirni Livewebcam] Creating a local Python environment...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv "%VENV_DIR%"
  ) else (
    python -m venv "%VENV_DIR%"
  )
  if errorlevel 1 goto :error
)

"%VENV_PY%" -c "import cv2, mediapipe, numpy, pyvirtualcam" >nul 2>nul
if errorlevel 1 (
  echo [Nirni Livewebcam] Installing the required packages. This needs internet and only happens once.
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)

echo [Nirni Livewebcam] Starting the webcam...
"%VENV_PY%" nirni_livewebcam.py --camera 1 %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Nirni Livewebcam could not start. See README.md for common fixes.
pause
exit /b 1
