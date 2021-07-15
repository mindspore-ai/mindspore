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
echo "bash run_eval.sh [NETWORK] [TRAIN_DATA_DIR] [TEST_DATA_DIR] [LOAD_CKPT_PATH] [LOG_PATH]"
echo "=============================================================================================================="

PROJECT_DIR=$(
  cd "$(dirname "$0")" || exit
  pwd
)

if [ ! -d $2 ]
then
  echo "error: TRAIN_DATA_DIR=$2 is not a directory"
  exit 1
fi

if [ ! -d $3 ]
then
  echo "error: TEST_DATA_DIR=$3 is not a directory"
  exit 1
fi

if [ ! -f $4 ]
then
  echo "error: LOAD_CKPT_PATH=$4 is not a file"
  exit 1
fi

if [ ! -d $5 ]
then
  echo "error: LOG_PATH=$5 is not a directory"
  exit 1
fi


python ${PROJECT_DIR}/../eval.py \
  --device_id=0 \
  --device_num=1 \
  --device_target=Ascend \
  --network=$1 \
  --train_data_dir=$2 \
  --test_data_dir=$3 \
  --load_ckpt_path=$4 \
  --log_path=$5 &> log &
