#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [ $# != 1 ]; then
  echo "Usage: sh run_standalone_train.sh [TRAIN_DATA_DIR]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)

if [ ! -d $PATH1 ]; then
  echo "error: TRAIN_DATA_DIR=$PATH1 is not a directory"
  exit 1
fi


if [ -d "train" ]; then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp -r ../src ./train
cd ./train || exit

env >env.log

nohup python train.py \
      --batch_size 16 \
      --lr 1e-4 \
      --scale 2 \
      --task_id 0 \
      --dir_data $PATH1 \
      --epochs 1000 \
      --test_every 100 \
      --patch_size 48 > train.log 2>&1 &
