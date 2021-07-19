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

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: sh run_standalone_train_gpu.sh [MINDRECORD_FILE] [CUDA_VISIBLE_DEVICES] [PRETRAINED_BACKBONE]"
    echo "   or: sh run_standalone_train_gpu.sh [MINDRECORD_FILE] [CUDA_VISIBLE_DEVICES]"
exit 1
fi

current_exec_path=$(pwd)
echo ${current_exec_path}

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export RANK_SIZE=1
export CUDA_VISIBLE_DEVICES=$2

SCRIPT_NAME='train.py'

ulimit -c unlimited

echo 'start training'
export RANK_ID=0
rm -rf train_alone_gpu
mkdir train_alone_gpu
cd train_alone_gpu

if [ $# == 2 ]
then
  python ${dirname_path}/${SCRIPT_NAME} \
      --world_size=1 \
      --device_target='GPU' \
      --mindrecord_path=$1 > train.log  2>&1 &
else
  python ${dirname_path}/${SCRIPT_NAME} \
      --world_size=1 \
      --device_target='GPU' \
      --mindrecord_path=$1 \
      --pretrained=$3 > train.log  2>&1 &
fi
echo 'running'
