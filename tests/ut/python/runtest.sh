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

CURRPATH=$(cd $(dirname $0); pwd)
cd ${CURRPATH}
PROJECT_PATH=${CURRPATH}/../../..
if [ $BUILD_PATH ];then
	echo "BUILD_PATH = $BUILD_PATH"
else
    BUILD_PATH=${PROJECT_PATH}/build
	echo "BUILD_PATH = $BUILD_PATH"
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BUILD_PATH}/third_party/gtest/lib
export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}:${PROJECT_PATH}/tests/ut/cpp/python_input:${PROJECT_PATH}/tests/ut/python
echo "export PYTHONPATH=$PYTHONPATH"

IGNORE_EXEC=""
if [ "x${ENABLE_GE}" == "xON" -o "x${ENABLE_GE}" == "xOn" -o "x${ENABLE_GE}" == "xon" -o \
     "x${ENABLE_GE}" == "xTrue" -o "x${ENABLE_GE}" == "xtrue" ]; then
    if [ $# -gt 0 ]; then
        IGNORE_EXEC="--ignore=$1/exec"
    else
        IGNORE_EXEC="--ignore=$CURRPATH/exec"
    fi
fi

if [ $# -gt 0 ]; then
    pytest -s --ignore=$1/pynative_mode $IGNORE_EXEC $1
else
    pytest --ignore=$CURRPATH/pynative_mode $IGNORE_EXEC $CURRPATH
fi

RET=$?
if [ "x${IGNORE_EXEC}" != "x" ]; then
    exit ${RET}
fi

if [ ${RET} -ne 0 ]; then
    exit ${RET}
fi

if [ $# -gt 0 ]; then
    pytest -s $1/pynative_mode
else
    pytest $CURRPATH/pynative_mode
fi

RET=$?
if [ ${RET} -ne 0 ]; then
    exit ${RET}
fi
