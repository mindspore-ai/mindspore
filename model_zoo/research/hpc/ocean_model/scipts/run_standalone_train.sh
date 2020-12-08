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

if [ $# != 5 ]
then
    echo "Usage: sh run_standalone_train_gpu.sh [im] [jm] [kb] [step] [DATASET_PATH]"
exit 1
fi


ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
mkdir ./outputs
echo "start training for device $DEVICE_ID"
env > env.log
python train.py --im=$1 --jm=$2 --kb=$3 --step=$4 --file_path=$5 --outputs_path="./outputs/" &> log &

cd ..
