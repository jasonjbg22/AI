@echo off
title ComfyUI Tailscale Web App

cd /d "%~dp0"

echo ==========================================
echo   ComfyUI Tailscale Web Interface
echo ==========================================
echo Using ComfyUI at 100.89.240.18:8188
echo.

python app.py

echo.
echo Server stopped.
pause
