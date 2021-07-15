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
echo "bash run_pretrain_gpu.sh [DATA_DIR] [SAVE_CHECKPOINT_PATH] [LOG_PATH]"
echo "=============================================================================================================="

PROJECT_DIR=$(
  cd "$(dirname "$0")" || exit
  pwd
)

if [ ! -d $1 ]
then
  echo "error: DATA_DIR=$1 is not a directory"
  exit 1
fi

if [ ! -d $2 ]
then
  echo "error: SAVE_CHECKPOINT_PATH=$2 is not a directory"
  exit 1
fi

if [ ! -d $3 ]
then
  echo "error: LOG_PATH=$3 is not a directory"
  exit 1
fi

python ${PROJECT_DIR}/../pretrain.py \
  --device_target=GPU \
  --data_dir=$1 \
  --save_checkpoint_path=$2 \
  --log_path=$3 &> log &
