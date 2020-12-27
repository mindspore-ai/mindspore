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
echo "Please run the scipt as: "
echo "sh test_gnmt_v2.sh \
  GNMT_ADDR RANK_TABLE_ADDR PRE_TRAIN_DATASET TEST_DATASET EXISTED_CKPT_PATH \
  VOCAB_ADDR BPE_CODE_ADDR TEST_TARGET"
echo "for example:"
echo "sh test_gnmt_v2.sh \
  /home/workspace/gnmt_v2 \
  /home/workspace/rank_table_8p.json \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.mindrecord \
  /home/workspace/dataset_menu/newstest2014.en.mindrecord \
  /home/workspace/gnmt_v2/gnmt-6_3452.ckpt \
  /home/workspace/wmt16_de_en/vocab.bpe.32000 \
  /home/workspace/wmt16_de_en/bpe.32000 \
  /home/workspace/wmt16_de_en/newstest2014.de"
echo "It is better to use absolute path."
echo "=============================================================================================================="

GNMT_ADDR=$1
RANK_TABLE_ADDR=$2
# train dataset addr
PRE_TRAIN_DATASET=$3
# eval dataset addr
TEST_DATASET=$4
EXISTED_CKPT_PATH=$5
VOCAB_ADDR=$6
BPE_CODE_ADDR=$7
TEST_TARGET=$8

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
    cp ${current_exec_path}/*.py .
    cp ${GNMT_ADDR}/*.py .
    cp -r ${GNMT_ADDR}/src .
    cp -r ${GNMT_ADDR}/config .
    export RANK_ID=$i
    export DEVICE_ID=$i
  python test_gnmt_v2.py \
    --config_train=${GNMT_ADDR}/config/config.json \
    --pre_train_dataset=$PRE_TRAIN_DATASET \
    --config_test=${GNMT_ADDR}/config/config_test.json \
    --test_dataset=$TEST_DATASET \
    --existed_ckpt=$EXISTED_CKPT_PATH \
    --vocab=$VOCAB_ADDR \
    --bpe_codes=$BPE_CODE_ADDR \
    --test_tgt=$TEST_TARGET > log_gnmt_network${i}.log 2>&1 &
    cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit
