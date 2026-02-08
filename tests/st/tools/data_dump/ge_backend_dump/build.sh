#!/usr/bin/env bash
set -e

unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64

echo "[INFO] Using C++ compiler:"
which c++
c++ --version

BUILD_DIR="build"
LIB_NAME="ms_debug_stub"

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

echo "[INFO] Configuring project with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

echo "[INFO] Building ${LIB_NAME} ..."
cmake --build . -- -j 4

echo "[INFO] Build completed."
echo "[INFO] Library output path:"
find . -type f -name "lib${LIB_NAME}.*" || true
