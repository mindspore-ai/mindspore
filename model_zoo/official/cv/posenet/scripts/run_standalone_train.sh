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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATASET_NAME DEVICE_ID"
echo "For example: bash run_standalone_train.sh dataset_name device_id"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

set -e

DATASET_NAME=$1
DEVICE_ID=$2
export DATASET_NAME
export DEVICE_ID

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf train
mkdir train
cd ./train
mkdir src
cd ../
cp ./train.py ./train
cp ./src/*.py ./train/src
cd ./train

env > env0.log

echo "Standalone train begin."
python train.py --run_distribute False \
                --device_id $2 --dataset $1 \
                --device_num 1 --is_modelarts False \
                --device_target "Ascend" > ./train_alone.log 2>&1 &

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
