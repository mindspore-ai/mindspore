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

if [ $# -lt 3 ]
then
    echo "Usage: sh run_train_cpu.sh [NET_NAME] [DATASET_NAME] [DATASET_PATH] [PRE_TRAINED](optional)"
    exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "../train" ];
then
    rm -rf ../train
fi
mkdir ../train
cd ../train || exit


if [ -f $4 ]  # pretrained ckpt
then
        python ${BASEPATH}/../train.py \
                --net=$1 \
                --dataset=$2 \
                --data_dir=$3 \
                --is_distributed=0 \
                --device_target='CPU' \
                --pretrained=$4 > train.log 2>&1 &
else
        python ${BASEPATH}/../train.py \
                --net=$1 \
                --dataset=$2 \
                --data_dir=$3 \
                --is_distributed=0 \
                --device_target='CPU' > train.log 2>&1 &
fi
