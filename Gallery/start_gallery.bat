@echo off
echo =========================================
echo Starting Neural Thumbs Virtual Server...
echo =========================================
echo.
echo Installing requirements (Flask)...
pip install flask

echo.
echo Launching Server...
python thumb_server.py
pause
