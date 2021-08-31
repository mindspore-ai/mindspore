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

if [ $# != 3 ] && [ $# != 4 ]
then 
    echo "Usage: bash scripts/run_distribute_train_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)"
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

if [ $# == 4 ]
then 
    PATH2=$(get_real_path $4)
fi


if [ ! -d $PATH1 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
    exit 1
fi 

if [ $# == 4 ] && [ ! -f $PATH2 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH2 is not a file"
    exit 1
fi

ulimit -u unlimited
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RANK_SIZE=8
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

TRAIN_OUTPUT=$BASE_PATH/train_parallel_$1_$2
if [ -d $TRAIN_OUTPUT ]; then
  rm -rf $TRAIN_OUTPUT
fi
mkdir $TRAIN_OUTPUT
cp ./train.py $TRAIN_OUTPUT
cp -r ./src $TRAIN_OUTPUT
cp -r ./model_utils $TRAIN_OUTPUT
cp $CONFIG_FILE $TRAIN_OUTPUT
cd $TRAIN_OUTPUT || exit

if [ $# == 3 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --net_name=$1 --dataset=$2 --run_distribute=True --output_path='./output'\
    --device_target="GPU" --data_path=$PATH1 \
    --config_path=${CONFIG_FILE##*/} &> log &
fi
    
if [ $# == 4 ]
then
    mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --net_name=$1 --dataset=$2 --run_distribute=True --output_path='./output'\
    --device_target="GPU" --data_path=$PATH1 --pre_trained=$PATH2 \
    --config_path=${CONFIG_FILE##*/} &> log &
fi
cd ..