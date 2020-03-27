#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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
set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
export DEVICE_NUM=2
export RANK_SIZE=$DEVICE_NUM
ulimit -n 65535
export DISTRIBUTION_FILE=$BASE_PATH/tdt${DEVICE_NUM}p/tdt_
export MINDSPORE_HCCL_CONFIG_PATH=$BASE_PATH/hccl_${DEVICE_NUM}p.json

for((i=0;i<$DEVICE_NUM;i++))
do
    rm -rf ./dataparallel$i
    mkdir ./dataparallel$i
    cp  *.py ./dataparallel$i
    cp -r kernel_meta ./dataparallel$i
    cd ./dataparallel$i
    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python resnet_cifar.py --run_distribute=1 --device_num=$DEVICE_NUM --epoch_size=10 >log 2>&1 &
    cd ../
done