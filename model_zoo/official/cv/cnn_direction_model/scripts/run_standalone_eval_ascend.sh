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
    echo "Usage: sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH]"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=4
export RANK_ID=0
export RANK_SIZE=1


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ ! -f $PATH2 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH2 is not a file"
exit 1
fi

if [ -d "eval" ];
then
    rm -rf ./eval
fi

mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
echo "start evaluation for device $DEVICE_ID"
env > env.log

python eval.py --dataset_path=$PATH1 --checkpoint_path=$PATH2 > eval.log 2>&1 &

cd ..
