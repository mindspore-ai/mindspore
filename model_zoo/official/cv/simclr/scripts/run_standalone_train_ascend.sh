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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 3 ]
then
    echo "Usage: sh run_standalone_train_ascend.sh [cifar10] [TRAIN_DATASET_PATH] [DEVICE_ID]"
exit 1
fi

script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
export DATASET_NAME=$1
export TRAIN_DATASET_PATH=$2
export DEVICE_ID=$3

python ${self_path}/../train.py --dataset_name=$DATASET_NAME --train_dataset_path=$TRAIN_DATASET_PATH \
               --device_id=$DEVICE_ID --device_target="Ascend" \
               --run_cloudbrain=False --run_distribute=False > log 2>&1 &