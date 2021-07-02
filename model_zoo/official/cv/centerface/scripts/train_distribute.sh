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
    echo "Usage: sh train_distribute.sh [RANK_TABLE] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS] [IMAGES]"
    echo "   or: sh train_distribute.sh [RANK_TABLE] [PRETRAINED_BACKBONE] [DATASET] [ANNOTATIONS]"
    echo "   or: sh train_distribute.sh [RANK_TABLE] [PRETRAINED_BACKBONE] [DATASET]"
    echo "   or: sh train_distribute.sh [RANK_TABLE] [PRETRAINED_BACKBONE]"
    echo "   or: sh train_distribute.sh [RANK_TABLE]"
    echo "   or: sh train_distribute.sh "
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

rm -rf ${current_exec_path}/device*
SCRIPT_NAME='train.py'

ulimit -c unlimited

root=${current_exec_path} # your script path
pretrained_backbone=${dirname_path}/mobilenet_v2.ckpt # or mobilenet_v2-b0353104.ckpt
dataset_path=$root/dataset/centerface
annot_path=$dataset_path/annotations/train.json
img_dir=$dataset_path/images/train/images
rank_table=$root/rank_table_8p.json

if [ $# -ge 1 ]
then
    rank_table=$(get_real_path $1)
    if [ ! -f $rank_table ]
    then
        echo "error: rank_table=$rank_table is not a file"
    exit 1
    fi
fi

if [ $# -ge 2 ]
then
    pretrained_backbone=$(get_real_path $2)
    if [ ! -f $pretrained_backbone ]
    then
        echo "error: pretrained_backbone=$pretrained_backbone is not a file"
    exit 1
    fi
fi

if [ $# -ge 3 ]
then
    dataset_path=$(get_real_path $3)
    if [ ! -d $dataset_path ]
    then
        echo "error: dataset_path=$dataset_path is not a dir"
    exit 1
    fi
fi

if [ $# -ge 4 ]
then
    annot_path=$(get_real_path $4)
    if [ ! -f $annot_path ]
    then
        echo "error: annot_path=$annot_path is not a file"
    exit 1
    fi
fi

if [ $# -ge 5 ]
then
    img_dir=$(get_real_path $5)
    if [ ! -d $img_dir ]
    then
        echo "error: img_dir=$img_dir is not a dir"
    exit 1
    fi
fi

echo $rank_table
echo $pretrained_backbone
echo $dataset_path
echo $annot_path
echo $img_dir

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export RANK_TABLE_FILE=$rank_table
export RANK_SIZE=8

cpus=`cat /proc/cpuinfo | grep "processor" | wc -l`
task_set_core=`expr $cpus \/ $RANK_SIZE` # for taskset, task_set_core=total cpu number/RANK_SIZE
echo 'start training'
for((i=0;i<=$RANK_SIZE-1;i++));
do
    echo 'start rank '$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    export RANK_ID=$i
    dev=`expr $i + 0`
    export DEVICE_ID=$dev
    taskset -c $((i*task_set_core))-$(((i+1)*task_set_core-1)) python ${dirname_path}/${SCRIPT_NAME} \
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
        --data_dir=$dataset_path \
        --annot_path=$annot_path \
        --img_dir=$img_dir > train.log  2>&1 &
done

echo 'running'
