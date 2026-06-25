#!/bin/bash
set -e

if [ "$RAILWAY_SERVICE_NAME" = "dashboard" ]; then
  exec streamlit run shared/dashboard/app.py --server.port "$PORT" --server.address 0.0.0.0
else
  exec uvicorn shared.api.main:app --host 0.0.0.0 --port "$PORT"
fi
