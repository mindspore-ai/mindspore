#!/bin/bash
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

CURRPATH=$(cd "$(dirname $0)"; pwd)
PROJECT_PATH=${CURRPATH}/../..
if [ $BUILD_PATH ];then
    echo "BUILD_PATH = $BUILD_PATH"
else
    BUILD_PATH=${PROJECT_PATH}/build
    echo "BUILD_PATH = $BUILD_PATH"
fi
cd ${BUILD_PATH}/mindspore/tests/ut

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BUILD_PATH}/third_party/gtest/lib
export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}:${PROJECT_PATH}/tests/ut/python_input:${PROJECT_PATH}/tests/python
echo "export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}:${PROJECT_PATH}/tests/ut/python_input"

if [ $# -gt 0 ]; then
    pytest -s --ignore=$1/pynative_mode $1
else
    pytest $CURRPATH/train $CURRPATH/infer $CURRPATH/model $CURRPATH/core $CURRPATH/adapter $CURRPATH/custom_ops $CURRPATH/data $CURRPATH/distributed $CURRPATH/nn $CURRPATH/exec $CURRPATH/optimizer --ignore=$CURRPATH/custom_ops/test_cus_conv2d.py --ignore=$CURRPATH/core/test_tensor.py --ignore=$CURRPATH/model/test_vgg.py --ignore=$CURRPATH/engine/test_training.py --ignore=$CURRPATH/nn/test_dense.py --ignore=$CURRPATH/nn/test_embedding.py --ignore=$CURRPATH/nn/test_conv.py --ignore=$CURRPATH/nn/test_activation.py --ignore=$CURRPATH/core/test_tensor_py.py
fi

RET=$?
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
