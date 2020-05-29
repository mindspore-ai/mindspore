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
echo "sh run_distribute_pretrain.sh DEVICE_NUM EPOCH_SIZE DATA_PATH MINDSPORE_HCCL_CONFIG_PATH"
echo "for example: sh run_distribute_pretrain.sh 8 52 /path/ende-l128-mindrecord00 /path/hccl.json"
echo "It is better to use absolute path."
echo "=============================================================================================================="

rm -rf run_distribute_train
mkdir run_distribute_train
cd run_distribute_train || exit

EPOCH_SIZE=$2
DATA_PATH=$3

export MINDSPORE_HCCL_CONFIG_PATH=$4
export RANK_TABLE_FILE=$4
export RANK_SIZE=$1
export HCCL_FLAG=1
export DEPLOY_MODE=0

for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    export GE_USE_STATIC_MEMORY=1

    mkdir helper$i
    cp -rf ../src/ ../train.py ./helper$i
    cd ./helper$i || exit
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log
    python train.py  \
    --distribute="true" \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --device_num=$RANK_SIZE \
    --enable_save_ckpt="true" \
    --enable_lossscale="true" \
    --do_shuffle="true" \
    --enable_data_sink="false" \
    --checkpoint_path="" \
    --save_checkpoint_steps=2500 \
    --save_checkpoint_num=30 \
    --data_path=$DATA_PATH > log.txt 2>&1 &
    cd ../
done
cd ..