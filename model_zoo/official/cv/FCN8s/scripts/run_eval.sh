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
echo "Please run the scipt as: "
echo "sh run_distribute_eval.sh DEVICE_NUM RANK_TABLE_FILE DATASET CKPT_PATH"
echo "for example: sh run_eval.sh [RANK_TABLE_FILE] /path/to/dataset /path/to/ckpt device_id"
echo "It is better to use absolute path."
echo "================================================================================================================="


export DATA_PATH=$1
CKPT_PATH=$2
DEVICE_ID=$3

rm -rf eval
mkdir ./eval
cp ./*.py ./eval
cp -r ./src ./eval
cd ./eval || exit
echo "start testing"
env > env.log
python eval.py  \
--device_id=$DEVICE_ID  \
--data_path=$DATA_PATH   \
--ckpt_path=$CKPT_PATH #> log.txt 2>&1 &

cd ../

