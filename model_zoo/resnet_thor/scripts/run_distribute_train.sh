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

if [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH] [DATASET_PATH] [DEVICE_NUM]"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: DMINDSPORE_HCCL_CONFIG_PATH=$1 is not a file"
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: DATASET_PATH=$2 is not a directory"
exit 1
fi

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit

ulimit -u unlimited
export DEVICE_NUM=$3
export RANK_SIZE=$3
export MINDSPORE_HCCL_CONFIG_PATH=$1

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp *.py ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"

    env > env.log
    python train.py --do_train=True --run_distribute=True --device_num=$DEVICE_NUM --dataset_path=$2 > log 2>&1 &
    cd ..
done
