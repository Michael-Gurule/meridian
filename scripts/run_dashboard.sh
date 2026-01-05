#!/bin/bash

# Run MERIDIAN Dashboard

echo "Starting MERIDIAN Portfolio Optimization Dashboard..."
echo "=================================================="

cd "$(dirname "$0")/.."

streamlit run src/dashboard/app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false