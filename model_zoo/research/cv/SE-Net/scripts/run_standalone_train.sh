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

if [ $# != 3 ]
then 
    echo "Usage: sh run_standalone_train.sh [NET] [DATASET_NAME] [DATASET_PATH]"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1
export NET=$1
export DATASET=$2
export DATASET_PATH=$3


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
env > env.log
# python train.py --net=se-resnet50  --dataset=imagenet2012  --dataset_path=/data/imagenet/train/
python train.py --net=$NET --dataset=$DATASET --dataset_path=$DATASET_PATH &> log &
cd ..
