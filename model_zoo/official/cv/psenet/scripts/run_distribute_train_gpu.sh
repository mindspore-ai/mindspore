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

if [ $# != 1 ] && [ $# != 2 ]
then
  echo "Usage: sh run_distribute_train.sh [PRETRAINED_PATH] [TRAIN_ROOT_DIR]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ ! -z "${1}" ];
then
  PATH1=$(get_real_path $1)
else
  PATH1=$1
fi

PATH2=$(get_real_path $2)
if [ ! -d $PATH2 ]
then
    echo "error: TRAIN_ROOT_DIR=$PATH2 is not a directory"
exit 1
fi

ulimit -u unlimited

export DEVICE_NUM=8
export RANK_SIZE=8

if [ -d "train_parallel" ];
then
  rm -rf ./train_parallel
fi
mkdir ./train_parallel
cp ./*.py ./train_parallel
cp ./scripts/*.sh ./train_parallel
cp -r ./src ./train_parallel
cp ./*yaml ./train_parallel
cd ./train_parallel || exit
env > env.log
if [ -f $PATH1 ]
then
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --run_distribute=True --device_target="GPU" --pre_trained=$PATH1 --TRAIN_ROOT_DIR=$PATH2 > log 2>&1 &
else
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --device_target="GPU" --run_distribute=True --TRAIN_ROOT_DIR=$PATH2 > log 2>&1 &
fi
cd .. || exit
