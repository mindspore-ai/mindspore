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
if [ $# != 2 ]
then
    echo "Usage: sh run_standalone_train_gpu.sh [cifar10|imagenet] [DATA_PATH]"
exit 1
fi

export DATASET_NAME=$1
export DATA_PATH=$2

python train.py --dataset_name=$DATASET_NAME --data_path=$DATA_PATH \
               --device_target="GPU" > log 2>&1 &
