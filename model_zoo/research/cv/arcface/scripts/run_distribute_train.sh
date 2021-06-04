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
echo "bash run.sh RANK_SIZE DATA_PATH"
echo "For example: bash run.sh 8 path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

RANK_SIZE=$1
DATA_PATH=$2

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
    echo "$RANK_TABLE_FILE"
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

for((i=0;i<RANK_SIZE;i++))
do
    rm -rf device$i
    mkdir device$i
    cp -r ./src/ ./device$i
    cp train.py  ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py  \
    --data_url $DATA_PATH \
    --device_num $RANK_SIZE \
    > train.log$i 2>&1 &
    cd ../
done
echo "finish"
cd ../
