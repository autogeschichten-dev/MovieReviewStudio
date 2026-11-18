@echo off
setlocal
cd /d "%~dp0"

where ffmpeg >nul 2>&1
if errorlevel 1 goto missing_ffmpeg
where ffprobe >nul 2>&1
if errorlevel 1 goto missing_ffmpeg

if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv ".venv"
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
exit /b %errorlevel%

:missing_ffmpeg
echo Install a Windows FFmpeg build with libx264 and put ffmpeg.exe and ffprobe.exe on PATH.
echo See README.md for setup instructions.
exit /b 1
