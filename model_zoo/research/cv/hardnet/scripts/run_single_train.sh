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
echo "bash run_standalone_train.sh DATA_PATH DEVICE_ID PRETRAINED_PATH"
echo "For example: bash run_single_train.sh /path/dataset 0 /path/pretrained_path"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
DEVICE_ID=$2
export DATA_PATH=${DATA_PATH}

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
export DEVICE_ID=$2
export RANK_ID=$2
env > env0.log
python3 train.py --dataset_path $1 --isModelArts False  --distribute False --device_id $2 --pre_ckpt_path $3 > train.log 2>&1

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
