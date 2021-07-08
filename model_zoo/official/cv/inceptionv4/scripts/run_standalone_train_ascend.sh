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

export RANK_SIZE=1
export DEVICE_ID=$1
DATA_DIR=$2
BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config.yaml"

rm -rf train_standalone
mkdir ./train_standalone
cd ./train_standalone || exit
echo  "start training for device id $DEVICE_ID"
env > env.log
python -u ../train.py --config_path=$CONFIG_FILE \
    --device_id=$1 \
    --dataset_path=$DATA_DIR > log.txt 2>&1 &
cd ../
