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
echo "bash run_distribute_train_ascend.sh RANK_TABLE_FILE DEVICE_NUM ACLIMDB_DIR GLOVE_DIR"
echo "for example: bash run_distribute_train_ascend.sh /path/hccl.json 8 /path/aclimdb /path/glove"
echo "It is better to use absolute path."
echo "=============================================================================================================="

ROOT_PATH=`pwd`
export RANK_TABLE_FILE=$1
RANK_SIZE=$2
ACLIMDB_DIR=$3
GLOVE_DIR=$4


for((i=0;i<${RANK_SIZE};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    cp ../../*.py ./
    cp -r ../../src ./
    export RANK_ID=$i
    export DEVICE_ID=$i
    python train.py  \
        --device_target="Ascend" \
        --aclimdb_path=$ACLIMDB_DIR \
        --glove_path=$GLOVE_DIR \
        --distribute=true \
        --device_num=$RANK_SIZE \
        --preprocess=true  \
        --preprocess_path=./preprocess > log.txt 2>&1 &
done
