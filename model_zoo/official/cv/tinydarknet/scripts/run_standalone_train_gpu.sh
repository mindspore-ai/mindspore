#!/usr/bin/env bash
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

echo "$1 $2 $3"

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [DEVICE_ID] [TRAIN_DATA_DIR] [cifar10|imagenet]"
exit 1
fi

expr $1 + 6 &>/dev/null
if [ $? != 0 ]
then
    echo "error:DEVICE_ID=$1 is not a integer"
exit 1
fi

if [ ! -d $2 ]
then
    echo "error:TRAIN_DATA_DIR=$2 is not a folder"
exit 1
fi
train_data_dir=$2
PROJECT_DIR=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="$PROJECT_DIR/../config/imagenet_config_gpu.yaml"
dataset_type='imagenet'
if [ $# == 3 ]
then
    if [ $3 != "cifar10" ] && [ $3 != "imagenet" ]
    then
        echo "error: the selected dataset is neither cifar10 nor imagenet"
    exit 1
    fi
    dataset_type=$3
fi

if [ $3 == 'imagenet' ]; then
  CONFIG_FILE="$PROJECT_DIR/../config/imagenet_config_gpu.yaml"
elif [ $3 == 'cifar10' ]; then
  CONFIG_FILE="$PROJECT_DIR/../config/cifar10_config_gpu.yaml"
else
  echo "error: the selected dataset is neither cifar10 nor imagenet"
exit 1
fi

export DEVICE_ID=$1
export RANK_ID=0
export DEVICE_NUM=1
export RANK_SIZE=1
rm -rf ./train_single_gpu
mkdir ./train_single_gpu
cp -r ../src ./train_single_gpu
cp ../train.py ./train_single_gpu
cp -r ../config ./train_single_gpu
echo "start training for rank $RANK_ID, device $DEVICE_ID, $dataset_type"
cd ./train_single_gpu || exit
python ./train.py --config_path=$CONFIG_FILE \
--dataset_name=$dataset_type --train_data_dir=$train_data_dir --device_target=GPU> ./train.log 2>&1 &

