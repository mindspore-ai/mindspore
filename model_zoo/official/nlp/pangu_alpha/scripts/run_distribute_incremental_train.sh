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
echo "Please run the script as: "
echo "bash run_distribute_train_incremental_train.sh DATA_DIR RANK_TABLE_FILE DEVICE_NUM"
echo "for example: scripts/run_distribute_incremental_train.sh DATASET RANK_TABLE RANK_SIZE PARAM_INIT_TYPE \\"
echo "MODE PER_BATCH STRATEGY_CKPT  CKPT_PATH CKPT_NAME"
echo "It is better to use absolute path."
echo "=============================================================================================================="

ROOT_PATH=`pwd`
DATA_DIR=$1
export RANK_TABLE_FILE=$2
RANK_SIZE=$3
PARAM_INIT_TYPE=$4
MODE=$5
PER_BATCH=$6
export STRATEGY=$7
export CKPT_PATH=$8
export CKPT_NAME=$9


for((i=0;i<${RANK_SIZE};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python ${ROOT_PATH}/train.py --distribute=true --device_num=$RANK_SIZE --data_url=$DATA_DIR --run_type=train \
    --param_init_type=$PARAM_INIT_TYPE --mode=$MODE --incremental_training=1 --strategy_load_ckpt_path=$STRATEGY \
    --load_ckpt_path=$CKPT_PATH --load_ckpt_name=$CKPT_NAME --per_batch_size=$PER_BATCH > log$i.log 2>&1 &
done
