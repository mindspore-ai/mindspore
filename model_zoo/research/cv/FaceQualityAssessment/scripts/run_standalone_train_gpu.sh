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

if [ $# -lt 1 ]
then
  echo "Usage: sh run_standalone_train_gpu.sh [TRAIN_LABEL_FILE] [PRETRAINED_BACKBONE](optional)"
  exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

current_exec_path=$(pwd)
rm -rf ${current_exec_path}/gpu
mkdir ${current_exec_path}/gpu
cd ${current_exec_path}/gpu || exit

if [ $2 ] # pretrained ckpt
then
  python ${BASEPATH}/../train.py \
            --train_label_file=$1 \
            --per_batch_size=256 \
            --device_target='GPU' \
            --pretrained=$2 > train.log  2>&1 &
else
  python ${BASEPATH}/../train.py \
            --train_label_file=$1 \
            --per_batch_size=256 \
            --device_target='GPU' > train.log  2>&1 &
fi
