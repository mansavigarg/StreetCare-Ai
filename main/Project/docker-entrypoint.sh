#!/bin/sh
set -eu

python run_api_service.py &
api_pid=$!

cleanup() {
    kill "$api_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

exec streamlit run run_dashboard.py \
    --server.address=0.0.0.0 \
    --server.port="${PORT:-8080}"
