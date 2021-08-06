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
    echo "Usage: bash train_distribute_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [PRETRAINED_BACKBONE] [ANNOTATIONS] [DATASET]"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

current_exec_path=$(pwd)
echo ${current_exec_path}

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}

rm -rf ${current_exec_path}/train_distribute_gpu
SCRIPT_NAME='train.py'

ulimit -c unlimited

if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$2"

pretrained_backbone=$(get_real_path $3)
if [ ! -f $pretrained_backbone ]
then
    echo "error: pretrained_backbone=$pretrained_backbone is not a file"
    exit 1
fi

annot_path=$(get_real_path $4)
if [ ! -f $annot_path ]
then
    echo "error: annot_path=$annot_path is not a file"
    exit 1
fi

dataset_path=$(get_real_path $5)
if [ ! -d $dataset_path ]
then
    echo "error: dataset_path=$dataset_path is not a dir"
    exit 1
fi

echo $pretrained_backbone
echo $annot_path
echo $dataset_path

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export RANK_SIZE=$1

echo "start training on $RANK_SIZE devices"

mkdir ${current_exec_path}/train_distribute_gpu
cd ${current_exec_path}/train_distribute_gpu || exit

mpirun -n $1 \
    python ${dirname_path}/${SCRIPT_NAME} \
    --lr=4e-3 \
    --per_batch_size=8 \
    --is_distributed=1 \
    --t_max=140 \
    --max_epoch=140 \
    --warmup_epochs=0 \
    --lr_scheduler=multistep \
    --lr_epochs=90,120 \
    --weight_decay=0.0000 \
    --loss_scale=1024 \
    --pretrained_backbone=$pretrained_backbone \
    --annot_path=$annot_path \
    --img_dir=$dataset_path \
    --device_target="GPU" > train.log  2>&1 &

echo 'running'
