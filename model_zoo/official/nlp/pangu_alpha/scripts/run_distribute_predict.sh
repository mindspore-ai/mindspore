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

execute_path=$(pwd)
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
export RANK_SIZE=$1
export RANK_TABLE_FILE=$2
export STRATEGY=$3
export TOKENIZER=$4
export CKPT_PATH=$5
export CKPT_NAME=$6
export MODE=$7
export PARAM_INIT_TYPE=$8

for((i=0;i<$RANK_SIZE;i++));
do
  rm -rf ${execute_path}/device_$i/
  mkdir ${execute_path}/device_$i/
  cd ${execute_path}/device_$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  python -s ${self_path}/../predict.py --strategy_load_ckpt_path=$STRATEGY --tokenizer_path=$TOKENIZER --load_ckpt_path=$CKPT_PATH \
                  --load_ckpt_name=$CKPT_NAME --mode=$MODE --run_type=predict --param_init_type=$PARAM_INIT_TYPE >log$i.log 2>&1 &
done