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

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [cifar10|imagenet2012]"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: DATA_PATH=$2 is not a directory"
exit 1
fi


dataset_type='cifar10'
if [ $# == 3 ]
then
    if [ $3 != "cifar10" ] && [ $3 != "imagenet2012" ]
    then
        echo "error: the selected dataset is neither cifar10 nor imagenet2012"
    exit 1
    fi
    dataset_type=$3
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$1

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
src_dir=$script_dir/..

start_idx=0
for((i=0;i<RANK_SIZE;i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end

    export DEVICE_ID=`expr $i \+ $start_idx`
    export RANK_ID=$i
    rm -rf ./train_parallel$DEVICE_ID
    mkdir ./train_parallel$DEVICE_ID
    cp $src_dir/*.py ./train_parallel$DEVICE_ID
    cp -r $src_dir/src ./train_parallel$DEVICE_ID
    cd ./train_parallel$DEVICE_ID || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID, $dataset_type"
    env > env.log
    taskset -c $cmdopt python train.py --data_path=$2 --device_target="Ascend" --device_id=$DEVICE_ID --is_distributed=1 --dataset=$dataset_type &> log &
    cd ..
done
