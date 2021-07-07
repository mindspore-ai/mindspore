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
echo "bash convert_dataset.sh DATA_FOLDER GLOVE_FILE TRAIN_DATA EVAL_DATA"
echo "for example:"
echo "bash convert_dataset.sh \
  /home/workspace/atae_lstm/data \
  /home/workspace/atae_lstm/data/glove.840B.300d.txt \
  /home/workspace/atae_lstm/data/train.mindrecord \
  /home/workspace/atae_lstm/data/test.mindrecord
  "
echo "It is better to use absolute path."
echo "=============================================================================================================="

DATA_FOLDER=$1
GLOVE_FILE=$2
TRAIN_DATA=$3
EVAL_DATA=$4

current_exec_path=$(pwd)
echo ${current_exec_path}

export GLOG_v=2

python create_dataset.py \
    --data_folder=$DATA_FOLDER \
    --glove_file=$GLOVE_FILE \
    --train_data=$TRAIN_DATA \
    --eval_data=$EVAL_DATA
