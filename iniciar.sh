#!/usr/bin/env bash
# Inicia el recomendador (ver README.md). Usa .venv si existe; si no, python3.
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
( sleep 3; command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8000 || open http://127.0.0.1:8000 ) >/dev/null 2>&1 &
exec "$PY" app.py "$@"
