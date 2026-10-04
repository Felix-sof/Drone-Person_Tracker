@echo off
rem Double-click to open the demo menu. Arguments pass through: Demo_Baslat.bat 1 --record
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=3
python run_demo.py %*
if errorlevel 1 pause
