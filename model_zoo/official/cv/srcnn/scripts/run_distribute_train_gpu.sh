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
    echo "Usage: sh run_distribute_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [PRE_TRAINED](optional)"
exit 1
fi

if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
exit 1
fi

export DEVICE_NUM=$1
export RANK_SIZE=$1

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "train_parallel" ];
then
    rm -rf train_parallel
fi
mkdir train_parallel
cd train_parallel || exit

export CUDA_VISIBLE_DEVICES="$2"

if [ -f $4 ]  # pre_trained ckpt
then 
    if [ $1 -gt 1 ]
    then
        mpirun -n $1 --allow-run-as-root python3 ${BASEPATH}/../train.py \
                                                 --dataset_path=$3 \
                                                 --run_distribute=True \
                                                 --pre_trained=$4 > log.txt 2>&1 &
    else
        python3 ${BASEPATH}/../train.py \
                --dataset_path=$3 \
                --pre_trained=$4 > log.txt 2>&1 &
    fi
else
    if [ $1 -gt 1 ]
    then
        mpirun -n $1 --allow-run-as-root python3 ${BASEPATH}/../train.py \
                                                 --run_distribute=True \
                                                 --dataset_path=$3 > log.txt 2>&1 &
    else
        python3 ${BASEPATH}/../train.py \
                --dataset_path=$3 > log.txt 2>&1 &
    fi
fi
