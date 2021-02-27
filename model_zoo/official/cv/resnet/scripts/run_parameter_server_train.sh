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
    echo "Usage: bash run_distribute_train.sh [resnet50|resnet101] [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)"
    exit 1
fi

if [ $1 != "resnet50" ] && [ $1 != "resnet101" ]
then 
    echo "error: the selected net is neither resnet50 nor resnet101"
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

export MS_SCHED_NUM=1
export MS_WORKER_NUM=$RANK_SIZE
export MS_SERVER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8081

export MS_ROLE=MS_SCHED
export DEVICE_ID=0
export RANK_ID=0
rm -rf ./sched
mkdir ./sched
cp ../*.py ./sched
cp *.sh ./sched
cp -r ../src ./sched
cd ./sched || exit
echo "start scheduler"
if [ $# == 4 ]
then
    python train.py --net=$1 --dataset=$2 --run_distribute=True --device_num=1 --dataset_path=$PATH2 --parameter_server=True &> sched.log &
fi

if [ $# == 5 ]
then
    python train.py --net=$1 --dataset=$2 --run_distribute=True --device_num=1 --dataset_path=$PATH2 --parameter_server=True --pre_trained=$PATH3 &> sched.log &
fi
cd ..

export MS_ROLE=MS_PSERVER
for((i=0; i<1; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./server_$i
    mkdir ./server_$i
    cp ../*.py ./server_$i
    cp *.sh ./server_$i
    cp -r ../src ./server_$i
    cd ./server_$i || exit
    echo "start server"
    if [ $# == 4 ]
    then
        python train.py --net=$1 --dataset=$2 --run_distribute=True --device_num=1 --dataset_path=$PATH2 --parameter_server=True &> server_$i.log &
    fi
    
    if [ $# == 5 ]
    then
        python train.py --net=$1 --dataset=$2 --run_distribute=True --device_num=1 --dataset_path=$PATH2 --parameter_server=True --pre_trained=$PATH3 &> server_$i.log &
    fi

    cd ..
done

export MS_ROLE=MS_WORKER
for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./worker_$i
    mkdir ./worker_$i
    cp ../*.py ./worker_$i
    cp *.sh ./worker_$i
    cp -r ../src ./worker_$i
    cd ./worker_$i || exit
    echo "start training for worker rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    if [ $# == 4 ]
    then
        python train.py --net=$1 --dataset=$2 --run_distribute=True --device_num=$DEVICE_NUM --dataset_path=$PATH2 --parameter_server=True &> worker_$i.log &
    fi
    
    if [ $# == 5 ]
    then
        python train.py --net=$1 --dataset=$2 --run_distribute=True --device_num=$DEVICE_NUM --dataset_path=$PATH2 --parameter_server=True --pre_trained=$PATH3 &> worker_$i.log &
    fi

    cd ..
done
