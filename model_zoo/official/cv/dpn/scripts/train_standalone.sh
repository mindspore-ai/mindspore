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
#Usage: sh train_standalone.sh [DEVICE_ID] [DATA_DIR] [SAVE_CKPT_PATH] [EVAL_EACH_EPOCH] [PATH_CHECKPOINT]!
export DEVICE_ID=$1
DATA_DIR=$2
SAVE_CKPT_PATH=$3
EVAL_EACH_EPOCH=$4

if [ $# == 5 ]
then
  PATH_CHECKPOINT=$5
fi

if [ $# == 4 ]
then
        python train.py  \
        --is_distributed=0 \
        --ckpt_path=$SAVE_CKPT_PATH\
        --eval_each_epoch=$EVAL_EACH_EPOCH\
        --data_dir=$DATA_DIR > train_log.txt 2>&1 &
    echo "    python train.py  \
        --is_distributed=0 \
        --ckpt_path=$SAVE_CKPT_PATH\
        --eval_each_epoch=$EVAL_EACH_EPOCH\
        --data_dir=$DATA_DIR > train_log.txt 2>&1 &"
fi
if [ $# == 5 ]
then
        python train.py  \
        --is_distributed=0 \
        --ckpt_path=$SAVE_CKPT_PATH\
        --pretrained=$PATH_CHECKPOINT \
        --data_dir=$DATA_DIR\
        --eval_each_epoch=$EVAL_EACH_EPOCH > train_log.txt 2>&1 &
fi