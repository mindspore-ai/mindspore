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

if [ $# != 4 ] && [ $# != 5 ]
then
  echo "Usage: bash run_distribute_train.sh [resnet18|resnet50|resnet101|se-resnet50] [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)"
  exit 1
fi

if [ $1 != "resnet18" ] && [ $1 != "resnet50" ] && [ $1 != "resnet101" ] && [ $1 != "se-resnet50" ]
then 
  echo "error: the selected net is neither resnet50 nor resnet101 and se-resnet50"
exit 1
fi

if [ $2 != "cifar10" ] && [ $2 != "imagenet2012" ]
then 
    echo "error: the selected dataset is neither cifar10 nor imagenet2012"
exit 1
fi

if [ $1 == "resnet101" ] && [ $2 == "cifar10" ]
then 
    echo "error: training resnet101 with cifar10 dataset is unsupported now!"
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

if [ $# == 5 ]
then 
    PATH3=$(get_real_path $5)
fi

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -d $PATH2 ]
then 
    echo "error: DATASET_PATH=$PATH2 is not a directory"
exit 1
fi 

if [ $# == 5 ] && [ ! -f $PATH3 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH3 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    if [ $# == 4 ]
    then    
        python train.py --net=$1 --dataset=$2 --run_distribute=True --device_num=$DEVICE_NUM --dataset_path=$PATH2 &> log &
    fi
    
    if [ $# == 5 ]
    then
        python train.py --net=$1 --dataset=$2 --run_distribute=True --device_num=$DEVICE_NUM --dataset_path=$PATH2 --pre_trained=$PATH3 &> log &
    fi

    cd ..
done
