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
echo "bash run.sh DATA_PATH EVAL_PATH CKPT_PATH RANK_SIZE"
echo "For example: bash run.sh /path/dataset /path/evalset /path/ckpt 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

export DEVICE_NUM=$4
export RANK_SIZE=$4
export DATASET_NAME=$1
export EVAL_PATH=$2
export CKPT_PATH=$3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

rm -rf ./train_parallel
mkdir ./train_parallel
cp -r ./src/ ./train_parallel
# shellcheck disable=SC2035
cp *.py ./train_parallel
cd ./train_parallel
env > env.log
echo "start training"
    mpirun -n $4 --allow-run-as-root \
    python train.py --device_num $4 --device_target GPU --data_url $1 \
    --ckpt_url $3 --eval_url $2 \
    --pretrained \
    > train.log 2>&1 &
