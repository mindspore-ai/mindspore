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

if [ $# != 2 ]  && [ $# != 3 ]
then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_standalone_train_gpu.sh [DATASET] [CONFIG_PATH] [DEVICE_ID](optional)"
  echo "for example: bash scripts/run_standalone_train_gpu.sh  /path/to/data/ /path/to/config/"
  echo "=============================================================================================================="
  exit 1
fi

if [ $# != 3 ]; then
  DEVICE_ID=0
else
  DEVICE_ID=`expr $3 + 0`
  if [ $? != 0 ]; then
    echo "DEVICE_ID=$3 is not an integer"
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
DATASET=$(get_real_path $1)
CONFIG_PATH=$(get_real_path $2)
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
TRAIN_OUTPUT=${PROJECT_DIR}/../train_standalone_gpu
if [ -d $TRAIN_OUTPUT ]; then
  rm -rf $TRAIN_OUTPUT
fi
mkdir $TRAIN_OUTPUT
cd $TRAIN_OUTPUT || exit
cp ../train.py ./
cp ../eval.py ./
cp -r ../src ./
cp $CONFIG_PATH ./
env > env.log
python train.py  --data_path=$DATASET  \
                 --config_path=${CONFIG_PATH##*/} \
                 --output ./output \
                 --device_target=GPU > train.log  2>&1 &
