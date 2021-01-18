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

echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "sh run_distribute_train_gpu.sh DEVICE_NUM EPOCH_SIZE LR DATASET PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "for example: sh run_distribute_train_gpu.sh 8 500 0.2 coco /opt/ssd-300.ckpt(optional) 200(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 4 ] && [ $# != 6 ]
then
    echo "Usage: sh run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] \
[PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

# Before start distribute train, first create mindrecord files.
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit
python train.py --only_create_dataset=True --run_platform="GPU" --dataset=$4

echo "After running the scipt, the network runs in the background. The log will be generated in LOG/log.txt"

export RANK_SIZE=$1
EPOCH_SIZE=$2
LR=$3
DATASET=$4
PRE_TRAINED=$5
PRE_TRAINED_EPOCH_SIZE=$6

rm -rf LOG
mkdir ./LOG
cp ./*.py ./LOG
cp -r ./src ./LOG
cd ./LOG || exit

if [ $# == 4 ]
then
    mpirun -allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py  \
    --distribute=True  \
    --lr=$LR \
    --dataset=$DATASET \
    --device_num=$RANK_SIZE  \
    --loss_scale=1 \
    --run_platform="GPU" \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
fi

if [ $# == 6 ]
then
    mpirun -allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py  \
    --distribute=True  \
    --lr=$LR \
    --dataset=$DATASET \
    --device_num=$RANK_SIZE  \
    --pre_trained=$PRE_TRAINED \
    --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE \
    --loss_scale=1 \
    --run_platform="GPU" \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
fi
