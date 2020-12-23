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
echo "sh run_standalone_train_ascend.sh PRE_TRAIN_DATASET"
echo "for example:"
echo "sh run_standalone_train_ascend.sh \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.mindrecord"
echo "It is better to use absolute path."
echo "=============================================================================================================="

PRE_TRAIN_DATASET=$1

export GLOG_v=2

current_exec_path=$(pwd)
echo ${current_exec_path}
if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp -r ../src ./train
cp -r ../config ./train
cd ./train || exit
echo "start for training"
env > env.log
python train.py \
  --config=${current_exec_path}/train/config/config.json \
  --pre_train_dataset=$PRE_TRAIN_DATASET > log_gnmt_network.log 2>&1 &
cd ..
