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

echo "======================================================================================================================================================="
echo "Please run the script as: "
echo "sh run_distribute_train.sh DEVICE_NUM EPOCH_SIZE MINDRECORD_DIR IMAGE_DIR ANNO_PATH RANK_TABLE_FILE PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "For example: sh run_distribute_train.sh 8 150 /data/Mindrecord_train /data /data/train.txt /data/hccl.json /opt/yolov3-150.ckpt(optional) 100(optional)"
echo "It is better to use absolute path."
echo "The learning rate is 0.005 as default, if you want other lr, please change the value in this script."
echo "======================================================================================================================================================="

if [ $# != 6 ] && [ $# != 8 ]
then
    echo "Usage: sh run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [MINDRECORD_DIR] [IMAGE_DIR] [ANNO_PATH] [RANK_TABLE_FILE] \
[PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

EPOCH_SIZE=$2
MINDRECORD_DIR=$3
IMAGE_DIR=$4
ANNO_PATH=$5
PRE_TRAINED=$7
PRE_TRAINED_EPOCH_SIZE=$8

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit

# Before start distribute train, first create mindrecord files.
python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --image_dir=$IMAGE_DIR  \
--anno_path=$ANNO_PATH
if [ $? -ne 0 ]
then
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export RANK_TABLE_FILE=$6
export RANK_SIZE=$1

for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i

    start=`expr $i \* 12`
    end=`expr $start \+ 11`
    cmdopt=$start"-"$end

    rm -rf LOG$i
    mkdir ./LOG$i
    cp  *.py ./LOG$i
    cp -r ./src ./LOG$i
    cd ./LOG$i || exit
    export RANK_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log

    if [ $# == 6 ]
    then
        taskset -c $cmdopt python train.py  \
        --distribute=True  \
        --lr=0.005 \
        --device_num=$RANK_SIZE  \
        --device_id=$DEVICE_ID  \
        --mindrecord_dir=$MINDRECORD_DIR  \
        --image_dir=$IMAGE_DIR  \
        --epoch_size=$EPOCH_SIZE  \
        --anno_path=$ANNO_PATH > log.txt 2>&1 &
    fi

    if [ $# == 8 ]
    then
        taskset -c $cmdopt python train.py  \
        --distribute=True  \
        --lr=0.005 \
        --device_num=$RANK_SIZE  \
        --device_id=$DEVICE_ID  \
        --mindrecord_dir=$MINDRECORD_DIR  \
        --image_dir=$IMAGE_DIR  \
        --epoch_size=$EPOCH_SIZE  \
        --pre_trained=$PRE_TRAINED \
        --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE \
        --anno_path=$ANNO_PATH > log.txt 2>&1 &
    fi

    cd ../
done
