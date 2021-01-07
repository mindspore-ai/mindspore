#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

export RANK_TABLE_FILE=$1
echo "RANK_TABLE_FILE=$RANK_TABLE_FILE"
export RANK_SIZE=$3
SAVE_PATH=$2

for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i

    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    echo "start training for rank $i, device $DEVICE_ID"

    cd ./train_parallel$i ||exit
    env > env.log
    cd ../
    python train.py  \
    --run-distribute \
    --ckpt-path=$SAVE_PATH  > train_parallel$i/log.txt 2>&1 &
    
    echo "python train.py  \
    --run-distribute \
    --ckpt-path=$SAVE_PATH  > train_parallel$i/log.txt 2>&1 &"

done
