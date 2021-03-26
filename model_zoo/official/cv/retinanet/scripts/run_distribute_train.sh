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
echo "sh run_distribute_train.sh DEVICE_NUM EPOCH_SIZE LR DATASET RANK_TABLE_FILE PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "for example: sh run_distribute_train.sh 8 500 0.1 coco /data/hccl.json /opt/retinanet-500_458.ckpt(optional) 200(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 4 ] && [ $# != 6 ]
then
    echo "Usage: sh run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] \
[RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

core_num=`cat /proc/cpuinfo |grep "processor"|wc -l`
process_cores=$(($core_num/8))

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export RANK_SIZE=$1
EPOCH_SIZE=$2
LR=$3
PRE_TRAINED=$5
PRE_TRAINED_EPOCH_SIZE=$6
export RANK_TABLE_FILE=$4

for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    rm -rf LOG$i
    mkdir ./LOG$i
    cp ./*.py ./LOG$i
    cp -r ./src ./LOG$i
    cp -r ./scripts ./LOG$i
    start=`expr $i \* $process_cores`
    end=`expr $start \+ $(($process_cores-1))`
    cmdopt=$start"-"$end
    cd ./LOG$i || exit
    export RANK_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log
    if [ $# == 4 ]
    then
        taskset -c $cmdopt python train.py  \
        --workers=$process_cores  \
        --distribute=True  \
        --lr=$LR \
        --device_num=$RANK_SIZE  \
        --device_id=$DEVICE_ID  \
        --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
    fi

    if [ $# == 6 ]
    then
        taskset -c $cmdopt python train.py  \
        --workers=$process_cores  \
        --distribute=True  \
        --lr=$LR \
        --device_num=$RANK_SIZE  \
        --device_id=$DEVICE_ID  \
        --pre_trained=$PRE_TRAINED \
        --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE \
        --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
    fi

    cd ../
done
