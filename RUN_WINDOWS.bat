@echo off
setlocal
if exist "%~dp0MovieReviewStudio.exe" (
    start "" "%~dp0MovieReviewStudio.exe"
    exit /b
)
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\launch-windows.ps1"
set "MRS_EXIT=%ERRORLEVEL%"
if not "%MRS_EXIT%"=="0" pause
exit /b %MRS_EXIT%
