#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

if [ $# != 1 ]
then 
    echo "Usage: sh run_standalone_train.sh [DATASET_PATH]"
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

if [ ! -d "$PATH1" ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp *.py ./train
cp *.sh ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
python train.py --do_train=True --dataset_path=$PATH1 &> log &
cd ..
