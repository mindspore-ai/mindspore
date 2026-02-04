#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
    if [ -d "build" ]; then
        echo "[CLEANUP] Removing 'build' directory..."
        rm -rf build
    fi
}

trap cleanup EXIT

echo "[INFO] Current working directory: $(pwd)"

if [ -d "build" ]; then
    echo "[INFO] Removing existing 'build' directory..."
    rm -rf build
fi

if [ ! -f "build.sh" ]; then
    echo "[ERROR] build.sh not found in $(pwd)" >&2
    exit 1
fi

echo "[INFO] Running build.sh..."
bash build.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] build.sh failed!" >&2
    exit 1
fi

if [ ! -f "ge_backend_add.py" ]; then
    echo "[ERROR] ge_backend_add.py not found!" >&2
    exit 1
fi

echo "[INFO] Running ge_backend_add.py..."
python ge_backend_add.py
test_exit_code=$?

if [ $test_exit_code -ne 0 ]; then
    echo "[ERROR] ge_backend_add.py exited with code $test_exit_code" >&2
    exit $test_exit_code
else
    echo "[SUCCESS] All steps completed successfully."
fi
