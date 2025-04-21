#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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

set -e

CURRPATH=$(cd "$(dirname $0)"; pwd)
PROJECT_PATH="${CURRPATH}/.."
PYTHONTEST_DIR="${PROJECT_PATH}/tests/perf_test"
PERF_RESULT_DIR="${CURRPATH}/"
PERF_SUFFIX=".perf"
if [[ "${BUILD_PATH}" ]];then
    echo "BUILD_PATH = ${BUILD_PATH}"
else
    BUILD_PATH="${PROJECT_PATH}/build"
    echo "BUILD_PATH = ${BUILD_PATH}"
fi

cd "${PROJECT_PATH}"; sh build.sh -t off -l none -r -p on -j 20; cd -

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${BUILD_PATH}/third_party/gtest/lib"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}:${PROJECT_PATH}/tests/ut/python_input"
echo "export PYTHONPATH=${PYTHONPATH}:${PROJECT_PATH}:${PROJECT_PATH}/tests/ut/python_input"

for f in "${PYTHONTEST_DIR}"/test_*.py
do
    target_file="${PERF_RESULT_DIR}$(basename ${f} .py)${PERF_SUFFIX}"
    pytest -s "${f}" > "${target_file}" 2>&1
done
