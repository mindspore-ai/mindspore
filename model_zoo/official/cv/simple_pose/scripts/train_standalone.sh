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
# Usage: train_standalone.sh [CKPT_SAVE_DIR] [DEVICE_ID]
echo "$1 $2 $3"

if [ $# != 1 ] && [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash train_standalone.sh [CKPT_SAVE_DIR] [DEVICE_ID] [BATCH_SIZE]"
exit 1
fi

DEVICE_ID=0

if [ $# -ge 2 ]
then
    expr $2 + 6 &>/dev/null
    if [ $? != 0 ]
    then
        echo "error:DEVICE_ID=$2 is not a integer"
    exit 1
    fi
    DEVICE_ID=$2
fi

BATCH_SIZE=128

if [ $# -ge 3 ]
then
    expr $3 + 6 &>/dev/null
    if [ $? != 0 ]
    then
        echo "error:BATCH_SIZE=$3 is not a integer"
    exit 1
    fi
    BATCH_SIZE=$3
fi

export DEVICE_ID=$DEVICE_ID

rm -rf ./train_single
mkdir ./train_single
echo "start training for rank 0, device $DEVICE_ID"
cd ./train_single ||exit
env >env.log
cd ../
python train.py \
    --ckpt_save_dir=$1 --batch_size=$BATCH_SIZE\
    > ./train_single/train_log.txt 2>&1 &
echo "    python train.py --ckpt_save_dir=$1 --batch_size=$BATCH_SIZE > ./train_single/train_log.txt 2>&1 &"
