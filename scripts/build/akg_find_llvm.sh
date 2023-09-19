#!/bin/bash
# Copyright 2021-2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Find a suitable LLVM version for AKG.
#
# This file generates a temporary cmake script file
# and executes it by `cmake -P` (cmake script mode).
#
# If no suitable LLVM is found, the `find_package` function runs normally,
# the `cmake` command exits with status `0`.
#
# If suitable LLVM is found, the `find_package` will encounter the error
# "add_library command is not scriptable" in `LLVMExports.cmake` of LLVM library.
# This error is caused because of running `cmake` in script mode.
# Finally the `cmake` command exit with status `1`.

echo "find_package(LLVM 16 QUIET)" > akg_llvm_tmp.cmake
echo "find_package(LLVM 15 QUIET)" >> akg_llvm_tmp.cmake
echo "find_package(LLVM 14 QUIET)" >> akg_llvm_tmp.cmake
echo "find_package(LLVM 13 QUIET)" >> akg_llvm_tmp.cmake
echo "find_package(LLVM 12 QUIET)" >> akg_llvm_tmp.cmake
cmake -P akg_llvm_tmp.cmake > /dev/null 2>&1
result=$?
rm akg_llvm_tmp.cmake

if [  ${result} -eq 0 ]; then
    echo "off"
else
    echo "on"
fi


