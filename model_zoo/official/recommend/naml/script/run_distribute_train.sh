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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train.sh [PLATFORM] [DEVICE_NUM] [DATASET] [DATASET_PATH] [RANK_TABLE_FILE]"
echo "for example: bash run_distribute_train.sh Ascend 8 large /path/MINDlarge /path/hccl_8p.json"
echo "It is better to use absolute path."
echo "=============================================================================================================="

PLATFORM=$1
export RANK_SIZE=$2
DATASET=$3
DATASET_PATH=$4
export RANK_TABLE_FILE=$5
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CHECKPOINT_PATH=${PROJECT_DIR}/checkpoint
cd ${PROJECT_DIR}/.. || exit
for((i=0;i<RANK_SIZE;i++))
do
    rm -rf LOG$i
    mkdir ./LOG$i
    cp ./*.py ./LOG$i
    cp -r ./src ./LOG$i
    cd ./LOG$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python train.py \
        --platform=${PLATFORM} \
        --device_num=${RANK_SIZE} \
        --device_id=${DEVICE_ID} \
        --dataset=${DATASET} \
        --dataset_path=${DATASET_PATH} \
        --save_checkpoint_path=${CHECKPOINT_PATH} \
        --weight_decay=False \
        --sink_mode=True > log.txt 2>&1 &
    cd ..
done

