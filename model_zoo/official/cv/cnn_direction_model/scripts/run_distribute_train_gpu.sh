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
    echo "Usage:
          sh scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
          "
    exit 1
fi

if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
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

export DEVICE_NUM=$1
export RANK_SIZE=$1

if [ -d "./train" ];
then
    rm -rf ./train
fi
mkdir ./train
cd ./train || exit

export CUDA_VISIBLE_DEVICES="$2"

if [ $# == 3 ]
then
    mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python ../train.py \
        --run_distribute=True \
        --train_dataset_path=$PATH1 > train.log 2>&1 &
fi

if [ $# == 4 ]
then
    mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python ../train.py \
        --run_distribute=True \
        --train_dataset_path=$PATH1 \
        --pre_trained=$PATH2 > train.log 2>&1 &
fi
