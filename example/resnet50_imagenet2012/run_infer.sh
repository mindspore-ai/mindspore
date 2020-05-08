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

if [ $# != 2 ]
then 
    echo "Usage: sh run_infer.sh [DATASET_PATH] [CHECKPOINT_PATH]"
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
PATH2=$(get_real_path $2)


if [ ! -d $PATH1 ]
then 
    echo "error: DATASET_PATH=$1 is not a directory"
exit 1
fi 

if [ ! -f $PATH2 ]
then 
    echo "error: CHECKPOINT_PATH=$2 is not a file"
exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "infer" ];
then
    rm -rf ./infer
fi
mkdir ./infer
cp *.py ./infer
cp *.sh ./infer
cd ./infer || exit
env > env.log
echo "start infering for device $DEVICE_ID"
python eval.py --do_eval=True --dataset_path=$PATH1 --checkpoint_path=$PATH2 &> log &
cd ..
