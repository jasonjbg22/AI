@echo off
echo ====================================================
echo   LAUNCHING AI SYNC ENGINE + NEURAL GALLERY
echo ====================================================
echo.

:: 1. Start the AI Sync Engine in a new minimized window
echo Starting AI Sync Engine (Background)...
start /min cmd /k "python ai_watcher.py"

:: 2. Start the Gallery Server
echo Starting Gallery Server (http://localhost:5001)...
cd Gallery
start cmd /k "python thumb_server.py"

echo.
echo ====================================================
echo   Both systems are starting. 
echo   Gallery: http://localhost:5001
echo ====================================================
echo.
pause
