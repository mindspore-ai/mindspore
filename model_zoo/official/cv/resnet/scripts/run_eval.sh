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

if [ $# != 4 ]
then 
    echo "Usage: bash run_eval.sh [resnet18|resnet50|resnet101|se-resnet50] [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]"
exit 1
fi

if [ $1 != "resnet18" ] && [ $1 != "resnet50" ] && [ $1 != "resnet101" ] && [ $1 != "se-resnet50" ]
then 
    echo "error: the selected net is neither resnet50 nor resnet101 nor se-resnet50"
exit 1
fi

if [ $2 != "cifar10" ] && [ $2 != "imagenet2012" ]
then 
    echo "error: the selected dataset is neither cifar10 nor imagenet2012"
exit 1
fi

if [ $1 == "resnet101" ] && [ $2 == "cifar10" ]
then
    echo "error: evaluating resnet101 with cifar10 dataset is unsupported now!"
exit 1
fi

if [ $1 == "se-resnet50" ] && [ $2 == "cifar10" ]
then
    echo "error: evaluating se-resnet50 with cifar10 dataset is unsupported now!"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $3)
PATH2=$(get_real_path $4)


if [ ! -d $PATH1 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi 

if [ ! -f $PATH2 ]
then 
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
env > env.log
echo "start evaluation for device $DEVICE_ID"
python eval.py --net=$1 --dataset=$2 --dataset_path=$PATH1 --checkpoint_path=$PATH2 &> log &
cd ..
