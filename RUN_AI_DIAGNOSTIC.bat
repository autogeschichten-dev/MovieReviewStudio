@echo off
setlocal
if not exist "%~dp0MovieReviewStudio.exe" (
    echo Extract the complete V3.1 Windows release ZIP before running this diagnostic.
    pause
    exit /b 1
)
set "MRS_REPORT=%TEMP%\MovieReviewStudio-AI-Diagnostic.txt"
start "" /wait "%~dp0MovieReviewStudio.exe" --ai-diagnostic "%MRS_REPORT%" "%~1"
set "MRS_RESULT=%ERRORLEVEL%"
if exist "%MRS_REPORT%" start "" notepad.exe "%MRS_REPORT%"
exit /b %MRS_RESULT%
