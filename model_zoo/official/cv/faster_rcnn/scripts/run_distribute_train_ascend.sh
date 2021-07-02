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

if [ $# -le 3 ]
then 
    echo "Usage: sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)"
exit 1
fi

if [ $3 != "resnet_v1_50" ] && [ $3 != "resnet_v1.5_50" ] && [ $3 != "resnet_v1_101" ] && [ $3 != "resnet_v1_152" ]
then 
  echo "error: the selected backbone must be resnet_v1_50, resnet_v1.5_50, resnet_v1_101, resnet_v1_152"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)
PATH3=$(get_real_path $4)
echo $PATH1
echo $PATH2
echo $PATH3

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -f $PATH2 ]
then 
    echo "error: PRETRAINED_PATH=$PATH2 is not a file"
exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: COCO_ROOT=$PATH3 is not a dir"
exit 1
fi

mindrecord_dir=$PATH3/MindRecord_COCO_TRAIN/
if [ $# -eq 5 ]
then
    mindrecord_dir=$(get_real_path $5)
    if [ ! -d $mindrecord_dir ]
    then
        echo "error: mindrecord_dir=$mindrecord_dir is not a dir"
    exit 1
    fi
fi
echo $mindrecord_dir

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
if [ $# -ge 1 ]; then
  if [ $3 == 'resnet_v1.5_50' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  elif [ $3 == 'resnet_v1_101' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_101.yaml"
  elif [ $3 == 'resnet_v1_152' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_152.yaml"
  elif [ $3 == 'resnet_v1_50' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  else
    echo "Unrecognized parameter"
    exit 1
  fi
else
  CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
fi

ulimit -u unlimited
export HCCL_CONNECT_TIMEOUT=600
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp ../*.yaml ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py --config_path=$CONFIG_FILE --coco_root=$PATH3 --mindrecord_dir=$mindrecord_dir --device_id=$i \
    --rank_id=$i --run_distribute=True --device_num=$DEVICE_NUM --pre_trained=$PATH2 --backbone=$3 &> log &
    cd ..
done