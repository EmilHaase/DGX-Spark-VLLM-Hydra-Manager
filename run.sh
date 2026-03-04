#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "==============================================================="
    echo "First time setup detected. Initializing environment..."
    echo "==============================================================="
    ./setup.sh
fi

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

source .venv/bin/activate

exec python3 main.py