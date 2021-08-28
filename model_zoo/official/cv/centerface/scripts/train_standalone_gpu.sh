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

if [ $# != 4 ]
then
    echo "Usage: bash train_standalone_gpu.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE] [ANNOTATIONS] [DATASET]"
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
echo "current_exec_path: "   ${current_exec_path}

dirname_path=$(dirname "$(pwd)")
echo "dirname_path: "   ${dirname_path}

SCRIPT_NAME='train.py'

ulimit -c unlimited

if [ $1 -lt 0 ] && [ $1 -gt 7 ]
then
    echo "error: DEVICE_ID=$1 is not in (0-7)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$1"

pretrained_backbone=$(get_real_path $2)
if [ ! -f $pretrained_backbone ]
then
    echo "error: pretrained_backbone=$pretrained_backbone is not a file"
    exit 1
fi

annot_path=$(get_real_path $3)
if [ ! -f $annot_path ]
then
    echo "error: annot_path=$annot_path is not a file"
    exit 1
fi

dataset_path=$(get_real_path $4)
if [ ! -d $dataset_path ]
then
    echo "error: dataset_path=$dataset_path is not a dir"
    exit 1
fi

echo $pretrained_backbone
echo $annot_path
echo $dataset_path

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export RANK_SIZE=1

echo 'start training'
rm -rf ${current_exec_path}/train_standalone_gpu
mkdir ${current_exec_path}/train_standalone_gpu
cd ${current_exec_path}/train_standalone_gpu || exit
export RANK_ID=0

python ${dirname_path}/${SCRIPT_NAME} \
    --lr=5e-4 \
    --per_batch_size=8 \
    --is_distributed=0 \
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
