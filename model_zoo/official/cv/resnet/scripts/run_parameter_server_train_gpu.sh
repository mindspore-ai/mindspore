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

if [ $# != 3 ] && [ $# != 4 ]
then 
    echo "Usage: bash run_distribute_train_gpu.sh [resnet50|resnet101] [cifar10|imagenet2012]  [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)"
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

if [ $# == 4 ]
then 
    PATH2=$(get_real_path $4)
fi


if [ ! -d $PATH2 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
    exit 1
fi 

if [ $# == 5 ] && [ ! -f $PATH2 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH2 is not a file"
    exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8

export MS_SCHED_NUM=1
export MS_WORKER_NUM=8
export MS_SERVER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8081

export MS_ROLE=MS_SCHED
rm -rf ./sched
mkdir ./sched
cp ../*.py ./sched
cp *.sh ./sched
cp -r ../src ./sched
cd ./sched || exit
if [ $# == 3 ]
then
    mpirun --allow-run-as-root -n 1 --output-filename log_output --merge-stderr-to-stdout \
    python train.py --net=$1 --dataset=$2 --run_distribute=True \
    --device_num=$DEVICE_NUM --device_target="GPU" --dataset_path=$PATH1 --parameter_server=True &> sched.log &
fi

if [ $# == 4 ]
then
    mpirun --allow-run-as-root -n 1 --output-filename log_output --merge-stderr-to-stdout \
    python train.py --net=$1 --dataset=$2 --run_distribute=True \
    --device_num=$DEVICE_NUM --device_target="GPU" --dataset_path=$PATH1 --parameter_server=True --pre_trained=$PATH2 &> sched.log &
fi
cd ..

export MS_ROLE=MS_PSERVER
for((i=0;i<$MS_SERVER_NUM;i++));
do
    rm -rf ./server_$i
    mkdir ./server_$i
    cp ../*.py ./server_$i
    cp *.sh ./server_$i
    cp -r ../src ./server_$i
    cd ./server_$i || exit
    if [ $# == 3 ]
    then
        mpirun --allow-run-as-root -n 1 --output-filename log_output --merge-stderr-to-stdout \
        python train.py --net=$1 --dataset=$2 --run_distribute=True \
        --device_num=$DEVICE_NUM --device_target="GPU" --dataset_path=$PATH1 --parameter_server=True &> server_$i.log &
    fi
        
    if [ $# == 4 ]
    then
        mpirun --allow-run-as-root -n 1 --output-filename log_output --merge-stderr-to-stdout \
        python train.py --net=$1 --dataset=$2 --run_distribute=True \
        --device_num=$DEVICE_NUM --device_target="GPU" --dataset_path=$PATH1 --parameter_server=True --pre_trained=$PATH2 &> server_$i.log &
    fi
    cd ..
done

export MS_ROLE=MS_WORKER
rm -rf ./worker
mkdir ./worker
cp ../*.py ./worker
cp *.sh ./worker
cp -r ../src ./worker
cd ./worker || exit
if [ $# == 3 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --net=$1 --dataset=$2 --run_distribute=True \
    --device_num=$DEVICE_NUM --device_target="GPU" --dataset_path=$PATH1 --parameter_server=True &> worker.log &
fi

if [ $# == 4 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --net=$1 --dataset=$2 --run_distribute=True \
    --device_num=$DEVICE_NUM --device_target="GPU" --dataset_path=$PATH1 --parameter_server=True --pre_trained=$PATH2 &> worker.log &
fi
cd ..
