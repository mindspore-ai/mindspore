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
echo "bash run_train.sh [PLATFORM] [DEVICE_ID] [DATASET] [DATASET_PATH]"
echo "for example: bash run_train.sh Ascend 0 large /path/MINDlarge"
echo "It is better to use absolute path."
echo "=============================================================================================================="

PLATFORM=$1
export RANK_ID=$2
export DEVICE_ID=$2
DATASET=$3
DATASET_PATH=$4
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CHECKPOINT_PATH="./checkpoint"

config_path="${PROJECT_DIR}/../MIND${DATASET}_config.yaml"
echo "config path is : ${config_path}"

python ${PROJECT_DIR}/../train.py \
    --config_path=${config_path} \
    --platform=${PLATFORM} \
    --dataset=${DATASET} \
    --dataset_path=${DATASET_PATH} \
    --save_checkpoint_path=${CHECKPOINT_PATH} \
    --weight_decay=False \
    --sink_mode=True

python ${PROJECT_DIR}/../eval.py \
    --config_path=${config_path} \
    --platform=${PLATFORM} \
    --dataset=${DATASET} \
    --dataset_path=${DATASET_PATH} \
    --checkpoint_path=${CHECKPOINT_PATH}/naml_last.ckpt
