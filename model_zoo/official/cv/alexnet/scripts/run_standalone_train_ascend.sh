#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ]
then
    echo "Usage: sh run_standalone_train_ascend.sh [cifar10|imagenet] [DATA_PATH] [DEVICE_ID] [CKPT_PATH]"
exit 1
fi

export DATASET_NAME=$1
export DATA_PATH=$2
export DEVICE_ID=$3
export CKPT_PATH=$4

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

if [ $# -ge 1 ]; then
  if [ $1 == 'imagenet' ]; then
    CONFIG_FILE="${BASE_PATH}/../config_imagenet.yaml"
  elif [ $1 == 'cifar10' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  else
    echo "Unrecognized parameter"
    exit 1
  fi
else
  CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
fi

python ../train.py --config_path=$CONFIG_FILE --dataset_name=$DATASET_NAME --data_path=$DATA_PATH \
--ckpt_path=$CKPT_PATH --device_id=$DEVICE_ID --device_target="Ascend" > log 2>&1 &
               
