@echo off
call "%~dp0run.bat" --virtual-camera %*
exit /b %errorlevel%
