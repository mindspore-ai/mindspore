#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
CURRPATH=$(dirname "${BASH_SOURCE-$0}")
PROJECT_PATH=$(cd ${CURRPATH}/../../.. || exit; pwd)

if [ $BUILD_PATH ];then
    echo "BUILD_PATH = $BUILD_PATH"
else
    BUILD_PATH=${PROJECT_PATH}/build
    echo "BUILD_PATH = $BUILD_PATH"
fi

MINDSPORE_PYTHON=${PROJECT_PATH}/mindspore/python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BUILD_PATH}/third_party/gtest/lib
export PYTHONPATH=$PYTHONPATH:${MINDSPORE_PYTHON}:${PROJECT_PATH}/tests/ut/cpp/python_input:${PROJECT_PATH}/tests/ut/python
echo "export PYTHONPATH=$PYTHONPATH"
export GC_COLLECT_IN_CELL=1