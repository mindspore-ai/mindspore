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
echo "bash run_distributed_train_gpu.sh PRE_TRAIN_DATASET"
echo "for example:"
echo "bash run_distributed_train_gpu.sh \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.mindrecord"
echo "It is better to use absolute path."
echo "=============================================================================================================="

PRE_TRAIN_DATASET=$1

current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_SIZE=8
export GLOG_v=2

rm -rf LOG
mkdir ./LOG
cp ../*.py ./LOG
cp ../*.yaml ./LOG
cp -r ../src ./LOG
cd ./LOG || exit
config_path="${current_exec_path}/LOG/default_config_gpu.yaml"
echo "config path is : ${config_path}"


if [ $# == 1 ]
then
    mpirun -allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python ../../train.py \
    --config_path=$config_path \
    --device_target="GPU" \
    --pre_train_dataset=$PRE_TRAIN_DATASET > log_gnmt_train.log 2>&1 &
fi

cd ${current_exec_path} || exit
