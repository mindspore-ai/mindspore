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
echo "bash run_distribute_train.sh DEVICE_NUM DEVICE_ID(0,1,2,3,4,5,6,7) DATA_PATH PRETRAINED_PATH"
echo "For example: sh run_distribute_train.sh 8 0,1,2,3,4,5,6,7 /path/dataset /path/pretrain_path"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$3
PRETRAINED_PATH=$4

if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
    exit 1
fi

export DEVICE_NUM=$1
export RANK_SIZE=$1
export CUDA_VISIBLE_DEVICES="$2"

cd ../
mpirun -n $1 --allow-run-as-root python3 train.py --device_target 'GPU' --isModelArts False --dataset_path ${DATA_PATH} --pre_ckpt_path ${PRETRAINED_PATH} > train.log 2>&1 &