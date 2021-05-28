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

DATA_DIR=$2
CKPT_PATH=$3
export RANK_TABLE_FILE=$1
export RANK_SIZE=8
export HCCL_CONNECT_TIMEOUT=600

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config.yaml"

cores=`cat /proc/cpuinfo|grep "processor" |wc -l`
echo "the number of logical core" $cores
avg_core_per_rank=`expr $cores \/ $RANK_SIZE`
core_gap=`expr $avg_core_per_rank \- 1`
echo "avg_core_per_rank" $avg_core_per_rank
echo "core_gap" $core_gap
for((i=0;i<RANK_SIZE;i++))
do
    start=`expr $i \* $avg_core_per_rank`
    export DEVICE_ID=$i
    export RANK_ID=$i
    export DEPLOY_MODE=0
    export GE_USE_STATIC_MEMORY=1
    end=`expr $start \+ $core_gap`
    cmdopt=$start"-"$end

    rm -rf train_parallel$i
    mkdir ./train_parallel$i
    cp  ../*.py ./train_parallel$i
    cp  ../*.yaml ./train_parallel$i
    cp  -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $i, device $DEVICE_ID"

    env > env.log
    taskset -c $cmdopt python ./train.py --config_path=$CONFIG_FILE \
    --is_distributed=True \
    --platform=Ascend \
    --dataset_path=$DATA_DIR --ckpt_path=$CKPT_PATH  > log.txt 2>&1 &
    cd ../
done
