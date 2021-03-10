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

DATA_DIR=$2
export RANK_TABLE_FILE=$1
export RANK_SIZE=8
export HCCL_CONNECT_TIMEOUT=600
echo "hccl connect time out has changed to 600 second" 
PATH_CHECKPOINT=""
if [ $# == 3 ]
then
    PATH_CHECKPOINT=$3
fi

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

    rm -rf LOG$i
    mkdir ./LOG$i
    cp  *.py ./LOG$i
    cd ./LOG$i || exit
    echo "start training for rank $i, device $DEVICE_ID"

    env > env.log
    taskset -c $cmdopt python ../train.py  \
    --is_distribute \
    --device_id=$DEVICE_ID \
    --pretrained=$PATH_CHECKPOINT \
    --data_dir=$DATA_DIR > log.txt 2>&1 &
    cd ../
done
