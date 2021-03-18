#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
echo "sh scripts/run_distribute_train.sh DEVICE_NUM RANK_TABLE_FILE NET_NAME DATASET_NAME DATASET CKPT_FILE"
echo "for example: sh scripts/run_distribute_train.sh 8 /data/hccl.json densenet121 imagenet /path/to/dataset ckpt_file"
echo "It is better to use absolute path."
echo "================================================================================================================="

echo "After running the script, the network runs in the background. The log will be generated in train_x/log.txt"

export RANK_SIZE=$1
export RANK_TABLE_FILE=$2
NET_NAME=$3
DATASET_NAME=$4
DATASET=$5
CKPT_FILE=$6

for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    rm -rf train_$i
    mkdir ./train_$i
    cp ./*.py ./train_$i
    cp -r ./src ./train_$i
    cd ./train_$i || exit
    export RANK_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log
    if [ -f $CKPT_FILE ]
    then
      python train.py --net=$NET_NAME --dataset=$DATASET_NAME --data_dir=$DATASET --pretrained=$CKPT_FILE > log.txt 2>&1 &
    else
      python train.py --net=$NET_NAME --dataset=$DATASET_NAME --data_dir=$DATASET > log.txt 2>&1 &
    fi

    cd ../
done
