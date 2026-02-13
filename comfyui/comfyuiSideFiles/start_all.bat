@echo off
title ComfyUI Suite Loader + Secure Tunnel
set PYTHON_EXECUTABLE=.\python_embeded\python.exe

:: 1. Force Sidecar dependencies again (just in case)
echo Checking dependencies...
"%PYTHON_EXECUTABLE%" -m pip install flask flask-cors --quiet

  2. Launch Sidecar
echo Starting Maintenance Sidecar (Port 8189)...
start /min "" "%PYTHON_EXECUTABLE%" maintenance.py


:: 4. Launch ComfyUI with AGGRESSIVE network flags
echo Starting ComfyUI (Port 8188)...
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"%PYTHON_EXECUTABLE%" -I ComfyUI\main.py --listen 0.0.0.0 --port 8188 --enable-cors-header "*" --preview-method auto --normalvram --fast

pause