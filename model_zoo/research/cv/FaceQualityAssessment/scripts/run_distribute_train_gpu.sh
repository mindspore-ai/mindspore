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
  echo "Usage: sh run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [TRAIN_LABEL_FILE]
        [PRETRAINED_BACKBONE](optional)"
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

current_exec_path=$(pwd)
rm -rf ${current_exec_path}/gpu_distributed
mkdir ${current_exec_path}/gpu_distributed
cd ${current_exec_path}/gpu_distributed || exit

export CUDA_VISIBLE_DEVICES="$2"

if [ $4 ] # pretrained ckpt
then
  if [ $1 -gt 1 ]
  then
    mpirun -n $1 --allow-run-as-root python ${BASEPATH}/../train.py \
                                                  --train_label_file=$3 \
                                                  --is_distributed=1 \
                                                  --per_batch_size=32 \
                                                  --device_target='GPU' \
                                                  --pretrained=$4 > train.log  2>&1 &
  else
    python ${BASEPATH}/../train.py \
            --train_label_file=$3 \
            --is_distributed=0 \
            --per_batch_size=256 \
            --device_target='GPU' \
            --pretrained=$4 > train.log  2>&1 &
  fi
else
  if [ $1 -gt 1 ]
  then
    mpirun -n $1 --allow-run-as-root python ${BASEPATH}/../train.py \
                                                  --train_label_file=$3 \
                                                  --is_distributed=1 \
                                                  --per_batch_size=32 \
                                                  --device_target='GPU' > train.log  2>&1 &
  else
    python ${BASEPATH}/../train.py \
            --train_label_file=$3 \
            --is_distributed=0 \
            --per_batch_size=256 \
            --device_target='GPU' > train.log  2>&1 &
  fi
fi
