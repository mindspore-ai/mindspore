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
#Usage: sh train_distributed.sh  [MINDSPORE_HCCL_CONFIG_PATH] [DATASET_PATH] [SAVE_CKPT_PATH] [RANK_SIZE] [EVAL_EACH_EPOCH] [PRETRAINED_CKPT_PATH](optional)

DATA_DIR=$2
export RANK_TABLE_FILE=$1
echo "RaNK_TABLE_FiLE=$RANK_TABLE_FILE"
export RANK_SIZE=$4
SAVE_PATH=$3
EVAL_EACH_EPOCH=$5
PATH_CHECKPOINT=""
if [ $# == 6 ]
then
    PATH_CHECKPOINT=$6
fi

device=(0 1 2 3 4 5 6 7)
for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=${device[$i]}
    export RANK_ID=$i

    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cp ./train.py ./train_parallel$i
    echo "start training for rank $i, device $DEVICE_ID"

    cd ./train_parallel$i ||exit
    env > env.log
    if [ $# == 5 ]
    then
        python train.py  \
        --is_distributed=1 \
        --ckpt_path=$SAVE_PATH \
        --eval_each_epoch=$EVAL_EACH_EPOCH\
        --data_dir=$DATA_DIR > log.txt 2>&1 &
        echo "python train.py  \
        --is_distributed=1 \
        --ckpt_path=$SAVE_PATH \
        --eval_each_epoch=$EVAL_EACH_EPOCH\
        --data_dir=$DATA_DIR > log.txt 2>&1 &"
    fi

    if [ $# == 6 ]
    then
        python train.py  \
        --is_distributed=1 \
        --eval_each_epoch=$EVAL_EACH_EPOCH\
        --ckpt_path=$SAVE_PATH \
        --pretrained=$PATH_CHECKPOINT \
        --data_dir=$DATA_DIR > log.txt 2>&1 &
    fi

    cd ../
done
