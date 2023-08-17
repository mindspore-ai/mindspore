#!/bin/bash
script_path=$(realpath "$(dirname $0)")

mkdir -p build_out
rm -rf build_out/*
cd build_out

opts="-DCMAKE_BUILD_TYPE=Release \
-DENABLE_SOURCE_PACKAGE=True \
-DENABLE_BINARY_PACKAGE=True \
-DASCEND_COMPUTE_UNIT=ascend910;ascend910b;ascend310p \
-DENABLE_TEST=False \
-Dvendor_name=mslite_ascendc \
-DASCEND_PYTHON_EXECUTABLE=python3 \
-DCMAKE_INSTALL_PREFIX=${script_path}/kernel_builder/ascend/ascendc/build_out \
-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/latest"

echo $opts
cmake .. $opts

target=package
if [ "$1"x != ""x ]; then target=$1; fi

cmake --build . --target $target -j16
if [ $? -ne 0 ]; then exit 1; fi
