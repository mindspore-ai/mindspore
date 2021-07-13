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

#Usage: sh run_distribute_train.sh [se-resnet50] [imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)


ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export NET=$1
export DATASET=$2
export DATASET_PATH=$3

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp *.sh ./train_parallel
cp -r ../src ./train_parallel
cd ./train_parallel || exit

echo "start distributed training with $DEVICE_NUM GPUs."

mpirun --allow-run-as-root -n $DEVICE_NUM \
    python train.py \
    --device_target="GPU" \
    --net=$NET \
    --dataset=$DATASET \
    --run_distribute=True \
    --device_num=$DEVICE_NUM \
    --dataset_path=$DATASET_PATH > log 2>&1 &
