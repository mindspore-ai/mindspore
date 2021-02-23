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
IGNORE_EXEC="--ignore=$CURRPATH/exec"
PROJECT_PATH=$(cd ${CURRPATH}/../../..; pwd)

if [ $BUILD_PATH ];then
    echo "BUILD_PATH = $BUILD_PATH"
else
    BUILD_PATH=${PROJECT_PATH}/build
    echo "BUILD_PATH = $BUILD_PATH"
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BUILD_PATH}/third_party/gtest/lib
export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}:${PROJECT_PATH}/tests/ut/cpp/python_input:${PROJECT_PATH}/tests/ut/python
echo "export PYTHONPATH=$PYTHONPATH"
export GC_COLLECT_IN_CELL=1

if [ $# -eq 1 ]  &&  ([ "$1" == "stage1" ] || [ "$1" == "stage2" ] || [ "$1" == "stage3" ]); then
    if [ $1 == "stage1" ]; then
        echo "run python dataset ut"
        pytest $CURRPATH/dataset

    elif [ $1 == "stage2" ]; then
        echo "run python parallel\train\ops ut"
        pytest -n 4 --dist=loadfile -v $CURRPATH/parallel $CURRPATH/train
        RET=$?
        if [ ${RET} -ne 0 ]; then
            exit ${RET}
        fi

        pytest -n 2 --dist=loadfile -v $CURRPATH/ops

    elif [ $1 == "stage3" ]; then
        echo "run other ut"
        pytest --ignore=$CURRPATH/dataset --ignore=$CURRPATH/parallel --ignore=$CURRPATH/train  --ignore=$CURRPATH/ops --ignore=$CURRPATH/pynative_mode $IGNORE_EXEC $CURRPATH
        RET=$?
        if [ ${RET} -ne 0 ]; then
            exit ${RET}
        fi

        pytest $CURRPATH/pynative_mode
    fi
else
    echo "run all python ut"
    pytest $CURRPATH/dataset
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest -n 4 --dist=loadfile -v $CURRPATH/parallel $CURRPATH/train
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest -n 2 --dist=loadfile -v $CURRPATH/ops
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest --ignore=$CURRPATH/dataset --ignore=$CURRPATH/parallel --ignore=$CURRPATH/train  --ignore=$CURRPATH/ops $IGNORE_EXEC --ignore=$CURRPATH/pynative_mode $CURRPATH
    RET=$?
    if [ ${RET} -ne 0 ]; then
        exit ${RET}
    fi

    pytest $CURRPATH/pynative_mode
fi

RET=$?
exit ${RET}
