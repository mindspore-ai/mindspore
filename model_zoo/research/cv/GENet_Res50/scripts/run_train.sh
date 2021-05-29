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

if [ $# != 5 ] && [ $# != 4 ]
then
  echo "Usage: bash run_train.sh [DATASET_PATH] [MLP] [EXTRA] [DEVICE_ID] [PRETRAINED_CKPT_PATH](optional)"
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)

if [ $# == 5 ]
then
    PATH2=$(get_real_path $5)
fi

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ $# == 5 ] && [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_FILE=$PATH2 is not a file"
exit 1
fi
ulimit -u unlimited

export DEVICE_NUM=1
export RANK_SIZE=1
export DEVICE_ID=$4
export RANK_ID=0

rm -rf ./train_standalone
mkdir ./train_standalone
cp ../*.py ./train_standalone
cp *.sh ./train_standalone
cp -r ../src ./train_standalone
cd ./train_standalone || exit
echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log
if [ $# == 4 ]
then
    python train.py --data_url=$PATH1 --mlp=$2 --extra=$3 &> log &
fi

if [ $# == 5 ]
then
    python train.py --data_url=$PATH1 --mlp=$2 --extra=$3 --pre_trained=$PATH2 &> log &
fi

cd ..
