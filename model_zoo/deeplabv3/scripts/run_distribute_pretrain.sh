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
echo "bash run_distribute_pretrain.sh DEVICE_NUM EPOCH_SIZE DATA_DIR MINDSPORE_HCCL_CONFIG_PATH"
echo "for example: bash run_distribute_train.sh 8 40 /path/zh-wiki/ /path/hccl.json"
echo "It is better to use absolute path."
echo "=============================================================================================================="
 
EPOCH_SIZE=$2
DATA_DIR=$3
 
export MINDSPORE_HCCL_CONFIG_PATH=$4
export RANK_TABLE_FILE=$4
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
    taskset -c $cmdopt python ../train.py  \
    --distribute="true" \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --enable_save_ckpt="true" \
    --checkpoint_url="/store1/deeplabv3/deeplabv3_split_url/train/checkpoint/CKP-12_732.ckpt" \
    --save_checkpoint_steps=10000 \
    --save_checkpoint_num=1 \
    --data_url=$DATA_DIR > log.txt 2>&1 &
    cd ../
done