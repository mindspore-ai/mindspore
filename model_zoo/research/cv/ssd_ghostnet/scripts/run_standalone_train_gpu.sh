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
echo "sh run_standalone_train.sh DEVICE_ID EPOCH_SIZE LR DATASET CONFIG_PATH PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "for example: sh run_standalone_train.sh 0 500 0.2 coco /config_path /opt/ssd-300.ckpt(optional) 200(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 5 ] && [ $# != 7 ]
then
    echo "Usage: sh run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] [CONFIG_PATH] \
    [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

# Before start distribute train, first create mindrecord files.
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit
python train.py --only_create_dataset=True  --device_target="GPU" --dataset=$4

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"
DEVICE_ID=$1
EPOCH_SIZE=$2
LR=$3
DATASET=$4
CONFIG_PATH=$5
PRE_TRAINED=$6
PRE_TRAINED_EPOCH_SIZE=$7

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
rm -rf LOG$DEVICE_ID
mkdir ./LOG$DEVICE_ID
cp ./*.py ./LOG$DEVICE_ID
cp -r ./src ./LOG$DEVICE_ID
cd ./LOG$DEVICE_ID || exit

echo "start training with device $DEVICE_ID"
env > env.log
if [ $# == 5 ]
then
    python train.py  \
    --lr=$LR \
    --dataset=$DATASET \
    --device_id=0  \
    --loss_scale=1 \
    --device_target="GPU" \
    --config_path=$CONFIG_PATH \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
fi

if [ $# == 7 ]
then
    python train.py  \
    --lr=$LR \
    --dataset=$DATASET \
    --device_id=0  \
    --loss_scale=1 \
    --device_target="GPU" \
    --pre_trained=$PRE_TRAINED \
    --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE \
    --config_path=$CONFIG_PATH \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
fi

cd ../
