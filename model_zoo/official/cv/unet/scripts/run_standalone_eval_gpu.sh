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
get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# != 3 ]  && [ $# != 4 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_standalone_eval_gpu.sh [DATASET] [CHECKPOINT] [CONFIG_PATH] [DEVICE_ID](optional)"
  echo "for example: bash run_standalone_eval_gpu.sh /path/to/data/ /path/to/checkpoint/ /path/to/config/"
  echo "=============================================================================================================="
  exit 1
fi

if [ $# != 4 ]; then
  DEVICE_ID=0
else
  DEVICE_ID=`expr $4 + 0`
  if [ $? != 0 ]; then
    echo "DEVICE_ID=$4 is not an integer"
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
DATASET=$(get_real_path $1)
CHECKPOINT=$(get_real_path $2)
CONFIG_PATH=$(get_real_path $3)
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
TRAIN_OUTPUT=${PROJECT_DIR}/../eval_gpu
if [ -d $TRAIN_OUTPUT ]; then
  rm -rf $TRAIN_OUTPUT
fi
mkdir $TRAIN_OUTPUT
cd $TRAIN_OUTPUT || exit
cp ../eval.py ./
cp -r ../src ./
cp $CONFIG_PATH ./
env > env.log
python eval.py   --data_path=$DATASET  \
                 --checkpoint_file_path=$CHECKPOINT \
                 --config_path=${CONFIG_PATH##*/} \
                 --device_target=GPU > eval.log  2>&1 &