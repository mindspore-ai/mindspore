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
echo "Please run the script as: "
echo "sh scripts/run_distribute_train.sh DEVICE_NUM DATASET_PATH RANK_TABLE_FILE"
echo "for example: sh scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json"

current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_SIZE=$1
data_path=$2
export RANK_TABLE_FILE=$3

for((i=0;i<=RANK_SIZE;i++));
do
    rm ${current_exec_path}/device_$i/ -rf
    mkdir ${current_exec_path}/device_$i
    cd ${current_exec_path}/device_$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python -u ${current_exec_path}/train.py  \
     --data_path $data_path \
     --dataset 'ml-1m'  \
     --train_epochs 50 \
     --output_path './output/' \
     --eval_file_name 'eval.log' \
     --loss_file_name 'loss.log'  \
     --checkpoint_path './checkpoint/' \
     --device_target="Ascend" \
     --device_id=$i \
     --is_distributed=1 \
     >log_$i.log 2>&1 &
done

