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

if [ $# != 4 ] && [ $# != 5 ]
then 
    echo "Usage: sh scripts/run_distribute_train.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINED_CKPT_PATH](optional)"
    exit 1
fi

if [ $1 != "squeezenet" ] && [ $1 != "squeezenet_residual" ]
then 
    echo "error: the selected net is neither squeezenet nor squeezenet_residual"
    exit 1
fi

if [ $2 != "cifar10" ] && [ $2 != "imagenet" ]
then 
    echo "error: the selected dataset is neither cifar10 nor imagenet"
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

BASE_PATH=$(dirname "$(dirname "$(readlink -f $0)")")
CONFIG_FILE="${BASE_PATH}/squeezenet_cifar10_config.yaml"

if [ $1 == "squeezenet" ] && [ $2 == "cifar10" ]; then
    CONFIG_FILE="${BASE_PATH}/squeezenet_cifar10_config.yaml"
elif [ $1 == "squeezenet" ] && [ $2 == "imagenet" ]; then
    CONFIG_FILE="${BASE_PATH}/squeezenet_imagenet_config.yaml"
elif [ $1 == "squeezenet_residual" ] && [ $2 == "cifar10" ]; then
    CONFIG_FILE="${BASE_PATH}/squeezenet_residual_cifar10_config.yaml"
elif [ $1 == "squeezenet_residual" ] && [ $2 == "imagenet" ]; then
    CONFIG_FILE="${BASE_PATH}/squeezenet_residual_imagenet_config.yaml"
else
     echo "error: the selected dataset is not in supported set{squeezenet, squeezenet_residual, cifar10, imagenet}"
exit 1
fi

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ./train.py ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cp -r ./model_utils ./train_parallel$i
    cp -r ./*.yaml ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    if [ $# == 4 ]
    then
        python train.py --net_name=$1 --dataset=$2 --run_distribute=True --device_num=$DEVICE_NUM --data_path=$PATH2 \
        --config_path=$CONFIG_FILE --output_path './output' &> log &
    fi
    
    if [ $# == 5 ]
    then
        python train.py --net_name=$1 --dataset=$2 --run_distribute=True --device_num=$DEVICE_NUM --data_path=$PATH2 \
        --config_path=$CONFIG_FILE --output_path './output' &> log &
    fi

    cd ..
done
