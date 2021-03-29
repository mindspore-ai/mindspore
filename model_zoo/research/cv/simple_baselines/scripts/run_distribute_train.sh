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
# Usage: sh train_distributed.sh  [MINDSPORE_HCCL_CONFIG_PATH] [SAVE_CKPT_PATH] [RANK_SIZE] 

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH RANK_SIZE"
echo "For example: bash run.sh /coco2017/train2017 2"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
RANK_SIZE=$2

EXEC_PATH=$(pwd)

echo "$EXEC_PATH"

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    mkdir src/utils
    cd ../
    cp ./train.py ./device$i
    cp ./src/config.py ./src/dataset.py ./src/pose_resnet.py ./src/network_with_loss.py  ./device$i/src
    cp ./src/utils/transforms.py ./device$i/src/utils
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py > train$i.log 2>&1 &
    echo "$i finish"
    cd ../
done

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
