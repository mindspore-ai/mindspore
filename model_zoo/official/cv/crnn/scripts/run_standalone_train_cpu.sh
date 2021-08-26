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

if [ $# != 2 ]; then
  echo "Usage: bash scripts/run_standalone_train_cpu.sh [DATASET_NAME] [DATASET_PATH]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_NAME=$1
PATH1=$(get_real_path $2)

if [ ! -d $PATH1 ]; then
  echo "error: DATASET_PATH=$PATH1 is not a directory"
  exit 1
fi

if [ -d "train" ]; then
    rm -rf ./train
fi
WORKDIR=./train_cpu
rm -rf $WORKDIR
mkdir $WORKDIR
cp ./*.py $WORKDIR
cp -r ./src $WORKDIR
cp ./*yaml $WORKDIR
cd $WORKDIR || exit

env >env.log
python train.py --train_dataset=$DATASET_NAME \
                --train_dataset_path=$PATH1 \
                --device_target=CPU \
                --model_version="V2" > log.txt 2>&1 &
cd ..
