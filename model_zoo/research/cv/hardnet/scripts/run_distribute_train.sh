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
echo "bash run_distribute_train.sh DATA_PATH pretrain_path RANK_SIZE"
echo "For example: bash run_distribute_train.sh /path/dataset /path/pretrain_path 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
PRETRAINED_PATH=$2
export DATA_PATH=${DATA_PATH}
RANK_SIZE=$3

EXEC_PATH=$(pwd)

echo "$EXEC_PATH"

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_4pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_4pcs.json
    export RANK_SIZE=4
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd ../
    cp ../*.py ./device$i
    cp ../src/*.py ./device$i/src
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python3 train.py --dataset_path ${DATA_PATH} --isModelArts False  --distribute True  --pre_ckpt_path ${PRETRAINED_PATH} > train$i.log 2>&1 &
    echo "$i finish"
    cd ../
done
rm -rf device0
mkdir device0
cd ./device0
mkdir src
cd ../
cp ../*.py ./device0
cp ../src/*.py ./device0/src
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
python3 train.py --dataset_path ${DATA_PATH} --isModelArts False  --distribute True --pre_ckpt_path ${PRETRAINED_PATH} > train0.log 2>&1

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
