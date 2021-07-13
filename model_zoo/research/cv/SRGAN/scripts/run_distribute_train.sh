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

if [ $# != 8 ]
then
    echo "Usage: sh run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [LRPATH] [GTPATH] [VGGCKPT] [VLRPATH] [VGTPATH]"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export RANK_SIZE=$1
export DISTRIBUTE=$2
export RANK_TABLE_FILE=$3
export LRPATH=$4
export GTPATH=$5
export VGGCKPT=$6
export VLRPATH=$7
export VGTPATH=$8



for((i=0;i<RANK_SIZE;i++))
do
        export DEVICE_ID=$i
        rm -rf ./train_parallel$i
        mkdir ./train_parallel$i
        cp -r ../src ./train_parallel$i
        cp -r ../*.py ./train_parallel$i
        cd ./train_parallel$i || exit
        export RANK_ID=$i
        echo "start training for rank $i, device $DEVICE_ID"
        env > env.log
        if [ $# == 8 ]
        then
                python train.py --run_distribute=$DISTRIBUTE --device_num=$RANK_SIZE \
                                --device_id=$DEVICE_ID --train_LR_path=$LRPATH --train_GT_path=$GTPATH --vgg_ckpt=$VGGCKPT \
                                --val_LR_path=$VLRPATH --val_GT_path=$VGTPATH &> log &
        fi
        cd ..
done
