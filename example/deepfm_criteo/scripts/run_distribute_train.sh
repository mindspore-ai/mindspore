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
echo "sh scripts/run_distribute_train.sh DEVICE_NUM DATASET_PATH MINDSPORE_HCCL_CONFIG_PAHT"
echo "for example: sh scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json"
echo "After running the script, the network runs in the background, The log will be generated in logx/output.log"


export RANK_SIZE=$1
DATA_URL=$2
export MINDSPORE_HCCL_CONFIG_PAHT=$3

for ((i=0; i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf log$i
    mkdir ./log$i
    cp *.py ./log$i
    cp -r src ./log$i
    cd ./log$i || exit
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log
    python -u train.py \
    --dataset_path=$DATA_URL \
    --ckpt_path="checkpoint" \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --do_eval=True > output.log 2>&1 &
    cd ../
done
