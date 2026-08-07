#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_DIR"

python3 -m streamlit run "$PROJECT_DIR/visualization/app.py"