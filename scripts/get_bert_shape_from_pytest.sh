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
SHP_BASENAME="test_bert_train"
BUILD_PATH="${PROJECT_PATH}/build"

cd "${PROJECT_PATH}"; sh build.sh -t off -l none -r -T; cd -

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${BUILD_PATH}/third_party/gtest/lib"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}:${PROJECT_PATH}/tests/ut/python_input"

test_bert_train="${PROJECT_PATH}/tests/perf_test/test_bert_train.py"

export SAVE_GRAPHS='YES'
export SAVE_GRAPHS_PATH="${PROJECT_PATH}"
for version in base large
do
    for batch_size in 1 2 4 8 16 32 64 128 256 512 1024
    do
        export VERSION="${version}"
        export BATCH_SIZE="${batch_size}"
        target_file="${PROJECT_PATH}/${SHP_BASENAME}.${VERSION}.${BATCH_SIZE}.shp"
        pytest "${test_bert_train}"
        cp "${SAVE_GRAPHS_PATH}/9_validate.dat" "${target_file}"
    done
done
