@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run dashboard/app.py --server.address=127.0.0.1 --server.port=8501 --server.fileWatcherType=none --browser.gatherUsageStats=false
if errorlevel 1 pause
