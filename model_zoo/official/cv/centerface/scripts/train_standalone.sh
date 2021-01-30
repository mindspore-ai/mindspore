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

if [ $# != 0 ] && [ $# != 1 ] && [ $# != 2 ] && [ $# != 3 ] && [ $# != 4 ] && [ $# != 5 ]
then
    echo "Usage: sh train_standalone.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]"
    echo "   or: sh train_standalone.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS]"
    echo "   or: sh train_standalone.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE] [DATASET]"
    echo "   or: sh train_standalone.sh [USE_DEVICE_ID] [PRETRAINED_BACKBONE]"
    echo "   or: sh train_standalone.sh [USE_DEVICE_ID]"
    echo "   or: sh train_standalone.sh "
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

root=${current_exec_path} # your script path
pretrained_backbone=${dirname_path}/mobilenet_v2.ckpt # or mobilenet_v2-b0353104.ckpt
dataset_path=$root/dataset/centerface
annot_path=$dataset_path/annotations/train.json
img_dir=$dataset_path/images/train/images
use_device_id=0

if [ $# == 1 ]
then
    use_device_id=$1
fi

if [ $# == 2 ]
then
    use_device_id=$1
    pretrained_backbone=$(get_real_path $2)
fi

if [ $# == 3 ]
then
    use_device_id=$1
    pretrained_backbone=$(get_real_path $2)
    dataset_path=$(get_real_path $3)
fi

if [ $# == 4 ]
then
    use_device_id=$1
    pretrained_backbone=$(get_real_path $2)
    dataset_path=$(get_real_path $3)
    annot_path=$(get_real_path $4)
fi

if [ $# == 5 ]
then
    use_device_id=$1
    pretrained_backbone=$(get_real_path $2)
    dataset_path=$(get_real_path $3)
    annot_path=$(get_real_path $4)
    img_dir=$(get_real_path $5)
fi

echo "use_device_id: "   $use_device_id
echo "pretrained_backbone: "   $pretrained_backbone
echo "dataset_path: "   $dataset_path
echo "annot_path: "   $annot_path
echo "img_dir: "   $img_dir

if [ ! -f $pretrained_backbone ]
then
    echo "error: pretrained_backbone=$pretrained_backbone is not a file"
exit 1
fi

if [ ! -d $dataset_path ]
then
    echo "error: dataset_path=$dataset_path is not a directory"
exit 1
fi

if [ ! -f $annot_path ]
then
    echo "error: annot_path=$annot_path is not a file"
exit 1
fi

if [ ! -d $img_dir ]
then
    echo "error: img_dir=$img_dir is not a directory"
exit 1
fi

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export RANK_SIZE=1

echo 'start training'
echo 'start rank '$use_device_id
rm -rf ${current_exec_path}/device$use_device_id
mkdir ${current_exec_path}/device$use_device_id
cd ${current_exec_path}/device$use_device_id || exit
export RANK_ID=0
dev=`expr $use_device_id + 0`
export DEVICE_ID=$dev
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
    --data_dir=$dataset_path \
    --annot_path=$annot_path \
    --img_dir=$img_dir > train.log  2>&1 &

echo 'running'
