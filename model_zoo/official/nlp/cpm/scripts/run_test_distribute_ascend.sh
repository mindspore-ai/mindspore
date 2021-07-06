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
if [ $# != 7 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_test_distribute_ascend.sh DEV_DATASET_PATH DEV_JSON_PATH TEST_DATASET_PATH TEST_JSON_PATH MODEL_CKPT CKPT_NUMBER RANK_TABLE_PATH"
echo "for example:"
echo "sh run_test_distribute_ascend.sh /disk0/dataset/finetune_dataset/finetune_dev.mindrecord /disk0/dataset/finetune_dataset/dev.json /disk0/dataset/finetune_dataset/finetune_test.mindrecord /disk0/dataset/finetune_dataset/test.json /disk2/ckpt_8p 8 /disk0/rank_table_2p.json"
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

DEV_DATASET=$(get_real_path $1)
echo $DEV_DATASET
DEV_LABEL=$(get_real_path $2)
echo $DEV_LABEL
TEST_DATASET=$(get_real_path $3)
echo $TEST_DATASET
TEST_LABEL=$(get_real_path $4)
echo $TEST_LABEL
MODEL_CKPT=$(get_real_path $5)
echo $MODEL_CKPT
CKPT_NUMBER=$6
echo $CKPT_NUMBER
RANK_TABLE_PATH=$(get_real_path $7)
echo $RANK_TABLE_PATH

current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_SIZE=2
export DEVICE_NUM=2
export RANK_TABLE_FILE=$RANK_TABLE_PATH

for((i=0;i<=1;i++));
do
    rm -rf ${current_exec_path}/eval$i
    mkdir ${current_exec_path}/eval$i
    cd ${current_exec_path}/eval$i || exit
    cp -r ../../*.py ./
    cp -r ../../src ./
    cp -r ../../scripts/*.sh ./

    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "start eval for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python ../../test.py  --dev_dataset $DEV_DATASET --dev_data_path $DEV_LABEL \
       --test_dataset $TEST_DATASET --test_data_path $TEST_LABEL \
       --ckpt_path_doc $MODEL_CKPT  --ckpt_partition $CKPT_NUMBER \
       --distribute True --has_train_strategy True --result_path ./result.txt > log_cpm.log 2>&1 &

    cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit
