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
echo "bash run.sh DEVICE_ID DATASET_NAME CKPT_PATH"
echo "For example: bash run_eval.sh device_id dataset ckpt_url"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

DEVICE_ID=$1
DATASET_NAME=$2
CKPT_PATH=$3
export DEVICE_ID
export DATASET_NAME
export CKPT_PATH

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf eval/
mkdir eval
cd ./eval
mkdir src
cd ../
cp ./eval.py ./eval
cp ./src/*.py ./eval/src
cd ./eval

env > env0.log
echo "Eval begin."
python eval.py --device_id $1 --dataset $2 --ckpt_url $3 --is_modelarts False --device_target "GPU" > ./eval.log 2>&1 &

if [ $? -eq 0 ];then
    echo "evaling success"
else
    echo "evaling failed"
    exit 2
fi
echo "finish"
cd ../
