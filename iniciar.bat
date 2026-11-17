@echo off
REM Inicia el recomendador (ver README.md). Usa .venv si existe; si no, el python del PATH.
cd /d "%~dp0"
set PY=python
if exist "%USERPROFILE%\.venvs\aw\python.exe" set PY="%USERPROFILE%\.venvs\aw\python.exe"
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
start "" http://127.0.0.1:8000
%PY% app.py %*
