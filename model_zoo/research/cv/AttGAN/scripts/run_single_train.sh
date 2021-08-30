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
    echo "Usage: sh run_single_train.sh [EXPERIMENT_NAME] [DATA_PATH] [ATTR_PATH]"
    exit 1
fi

experiment_name=$1
data_path=$2
attr_path=$3

cores=`cat /proc/cpuinfo|grep "processor" |wc -l`
echo "the number of logical core" $cores

export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

rm -rf LOG
mkdir ./LOG
cd ./LOG || exit
echo "Start training for rank 0, device 0, directory is LOG"

env > env.log
cd ../../

python train.py  \
--img_size 128 \
--shortcut_layers 1 \
--inject_layers 1 \
--experiment_name $experiment_name \
--data_path $data_path \
--attr_path $attr_path > ./scripts/LOG/log.txt 2>&1 &
