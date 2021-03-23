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
echo "bash run_distribute_pretrain.sh DEVICE_NUM EPOCH_SIZE DATA_DIR SCHEMA_DIR RANK_TABLE_FILE"
echo "for example: bash run_distribute_pretrain.sh 8 1 /path/zh-wiki/ /path/Schema.json /path/hccl.json"
echo "It is better to use absolute path."
echo "=============================================================================================================="

EPOCH_SIZE=$2
DATA_DIR=$3
SCHEMA_DIR=$4

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/ || exit

ulimit -u unlimited
export RANK_TABLE_FILE=$5
export RANK_SIZE=$1
export HCCL_CONNECT_TIMEOUT=300

for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$(( $i + 0 ))
    export RANK_ID=$i

    rm -rf LOG$i
    mkdir ./LOG$i
    cp  ../*.py ./LOG$i
    cp -r ../src ./LOG$i
    cd ./LOG$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python run_pretrain.py  \
    --distribute="true" \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --device_num=$RANK_SIZE \
    --enable_save_ckpt="true" \
    --enable_lossscale="false" \
    --do_shuffle="false" \
    --enable_data_sink="true" \
    --data_sink_steps=1000 \
    --load_checkpoint_path="" \
    --save_checkpoint_path='./' \
    --save_checkpoint_steps=1000 \
    --train_steps=3000 \
    --save_checkpoint_num=30 \
    --data_dir=$DATA_DIR \
    --schema_dir=$SCHEMA_DIR > log.txt 2>&1 &
    cd ../
done
