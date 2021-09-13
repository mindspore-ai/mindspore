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

if [ $# != 3 ]
then
  echo "Usage: bash run_standalone_train.sh [DATA_URL] [CKPT_URL] [MODELART]"
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
if [ $# == 3 ]
then
    PATH2=$(get_real_path $2)
fi

if [ ! -d $PATH1 ]
then
    echo "error: DATA_URL=$PATH1 is not a directory"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: CKPT_URL=$PATH2 is not a directory"
exit 1
fi

PATH3=$3

echo "$PATH1"
echo "$PATH2"
echo "$PATH3"

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=6
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train
echo "start training for device $DEVICE_ID"
env > env.log

if [ $# == 3 ]
then
    python train.py  --data_url=$PATH1 --ckpt_url=$PATH2 --modelart=$PATH3 &> train.log &
fi
cd ..
