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
ulimit -u unlimited
export DEVICE_ID=1
export RANK_SIZE=1
export RUN_OFFLINE=$1
export TRAIN_PATH=$2
export TRAIN_GT_PATH=$3
export VAL_PATH=$4
export VAL_GT_PATH=$5
export CKPT_PATH=$6

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.

if [ $# == 6 ]
then
    python train.py --run_offline=$RUN_OFFLINE --train_path=$TRAIN_PATH --train_gt_path=$TRAIN_GT_PATH \
                    --val_path=$VAL_PATH --val_gt_path=$VAL_GT_PATH --ckpt_path=$CKPT_PATH &> log &
fi
cd ..

