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
if [ $# != 5 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_distribute_train_ascend_multi_machine.sh DATASET_PATH CKPT_PATH RANK_TABLE_PATH SERVER_ID RANK_SIZE_ALL"
echo "for example:"
echo "sh run_distribute_train_ascend_multi_machine.sh /disk0/dataset/finetune_dataset/train.mindrecord /disk0/cpm_ckpt_ms/cpm_mindspore_1p_fp32.ckpt /disk0/rank_table_32p.json 0 32"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET=$(get_real_path $1)
echo $DATASET
PRECKPT=$(get_real_path $2)
RANK_TABLE_PATH=$(get_real_path $3)
SERVER_ID=$4
RANK_SIZE_ALL=$5

echo $DATANAME

current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_TABLE_FILE=$RANK_TABLE_PATH


echo $RANK_TABLE_FILE
export RANK_SIZE=$RANK_SIZE_ALL
export DEVICE_NUM=8

RANK_START=$(($DEVICE_NUM * $SERVER_ID))
GROUP_RANK=2
GROUP_DIFF=7

for((i=0;i<=3;i++));
do
    rm -rf ${current_exec_path}/device$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    cp ../../*.py ./
    cp -r ../../src ./
    cp -r ../*.sh ./
    export RANK_ID=$(((i*GROUP_RANK)+RANK_START))
    export DEVICE_ID=$i
    echo "start training for device $DEVICE_ID, rank $RANK_ID"
    python ../../train.py --dataset $DATASET --pretrain_ckpt_path $PRECKPT --multi_machine True > log_cpm.log 2>&1 &
    cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit


for((i=4;i<=7;i++));
do
    rm -rf ${current_exec_path}/device$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    cp ../../*.py ./
    cp -r ../../src ./
    cp -r ../*.sh ./
    export RANK_ID=$(((i*GROUP_RANK)-GROUP_DIFF+RANK_START))
    export DEVICE_ID=$i
    echo "start training for device $DEVICE_ID, rank $RANK_ID"
    python ../../train.py --dataset $DATASET --pretrain_ckpt_path $PRECKPT --multi_machine True > log_cpm.log 2>&1 &
    cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit
