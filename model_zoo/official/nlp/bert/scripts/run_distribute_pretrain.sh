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

echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "bash run_distribute_pretrain.sh DEVICE_NUM EPOCH_SIZE DATA_DIR SCHEMA_DIR MINDSPORE_HCCL_CONFIG_PATH"
echo "for example: bash run_distribute_pretrain.sh 8 40 /path/zh-wiki/ /path/Schema.json /path/hccl.json"
echo "It is better to use absolute path."
echo "=============================================================================================================="

EPOCH_SIZE=$2
DATA_DIR=$3
SCHEMA_DIR=$4
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export RANK_TABLE_FILE=$5
export RANK_SIZE=$1
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
    mkdir -p ms_log
    CUR_DIR=`pwd`
    export GLOG_log_dir=${CUR_DIR}/ms_log
    export GLOG_logtostderr=0
    env > env.log
    taskset -c $cmdopt python ${PROJECT_DIR}/../run_pretrain.py  \
    --distribute="true" \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --device_num=$RANK_SIZE \
    --enable_save_ckpt="true" \
    --enable_lossscale="true" \
    --do_shuffle="true" \
    --enable_data_sink="true" \
    --data_sink_steps=100 \
    --load_checkpoint_path="" \
    --save_checkpoint_steps=10000 \
    --save_checkpoint_num=1 \
    --data_dir=$DATA_DIR \
    --schema_dir=$SCHEMA_DIR > log.txt 2>&1 &
    cd ../
done
