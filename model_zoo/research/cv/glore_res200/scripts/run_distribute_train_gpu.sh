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
echo "bash run_distribute_train.sh DATA_PATH RANK_SIZE"
echo "For example: bash run_distribute_train.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DEVICE_NUM=$1
DATA_PATH=$2
export DATA_PATH=${DATA_PATH}
export DEVICE_NUM=$1
export RANK_SIZE=$1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf ./train_parallel
mkdir ./train_parallel
cd ./train_parallel
mkdir src
cd ../
cp *.py ./train_parallel
cp src/*.py ./train_parallel/src
cd ./train_parallel
env > env.log
echo "start training"
    mpirun -n $1 --allow-run-as-root \
           python3 train.py --device_num=$1 --data_url=$2 --isModelArts=False --run_distribute=True \
           --device_target="GPU" > train.log 2>&1 &

