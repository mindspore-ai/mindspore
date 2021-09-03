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

if [ $# != 5 ]
then 
    echo "Usage: bash scripts/run_eval_gpu.sh [squeezenet|squeezenet_residual] [cifar10|imagenet] [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]"
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

PATH1=$(get_real_path $4)
PATH2=$(get_real_path $5)


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

expr $3 + 6 &>/dev/null
if [ $? != 0 ]; then
  echo "DEVICE_ID=$3 is not an integer!"
exit 1
fi

ulimit -u unlimited
export CUDA_VISIBLE_DEVICES=$3

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

EVAL_OUTPUT=$BASE_PATH/eval_$3_$1_$2
if [ -d $EVAL_OUTPUT ];
then
    rm -rf $EVAL_OUTPUT
fi
mkdir $EVAL_OUTPUT
cp ./eval.py $EVAL_OUTPUT
cp -r ./src $EVAL_OUTPUT
cp -r ./model_utils $EVAL_OUTPUT
cp $CONFIG_FILE $EVAL_OUTPUT
cd $EVAL_OUTPUT || exit
env > env.log
echo "start evaluation for device $3"
python eval.py --net_name=$1 --dataset=$2 --data_path=$PATH1 --checkpoint_file_path=$PATH2 --device_target="GPU" \
--config_path=${CONFIG_FILE##*/} --output_path='./output' &> log &
cd ..
