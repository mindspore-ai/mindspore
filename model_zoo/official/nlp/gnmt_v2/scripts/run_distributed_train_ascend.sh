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
echo "sh run_distributed_train_ascend.sh RANK_TABLE_ADDR PRE_TRAIN_DATASET"
echo "for example:"
echo "sh run_distributed_train_ascend.sh \
  /home/workspace/rank_table_8p.json \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.mindrecord"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_TABLE_ADDR=$1
PRE_TRAIN_DATASET=$2

current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_TABLE_FILE=$RANK_TABLE_ADDR
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_ADDR

echo $RANK_TABLE_FILE
export RANK_SIZE=8
export GLOG_v=2

for((i=0;i<=7;i++));
do
    rm -rf ${current_exec_path}/device$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    cp ../../*.py .
    cp ../../*.yaml .
    cp -r ../../src .
    cp -r ../../model_utils .
    export RANK_ID=$i
    config_path="${current_exec_path}/device${i}/default_config.yaml"
    echo "config path is : ${config_path}"
    python ../../train.py \
      --config_path=$config_path \
      --pre_train_dataset=$PRE_TRAIN_DATASET \
      --device_id=$i > log_gnmt_network${i}.log 2>&1 &
      cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit
