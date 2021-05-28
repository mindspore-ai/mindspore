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
if [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train.sh [DATASET_PATH] [PRETRAINED_BACKBONE] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)
PRETRAINED_BACKBONE=$(get_real_path $2)
RANK_TABLE_FILE=$(get_real_path $3)
echo $DATASET_PATH
echo $PRETRAINED_BACKBONE
echo $RANK_TABLE_FILE

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

if [ ! -f $PRETRAINED_BACKBONE ]
then
    echo "error: PRETRAINED_PATH=$PRETRAINED_BACKBONE is not a file"
exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=$3

export RANK_ID=$3
export RANK_SIZE=1

rm -rf ./train_standalone
mkdir ./train_standalone
cp ../*.py ./train_standalone
cp -r ../src ./train_standalone
cd ./train_standalone || exit
echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log
python train.py \
    --data_dir=$DATASET_PATH \
    --pretrained_backbone=$PRETRAINED_BACKBONE \
    --device_id=$DEVICE_ID \
    --is_distributed=0 \
    --lr=0.001 \
    --max_epoch=600 \
    --per_batch_size=24 \
    --lr_scheduler=my_lr  > log.txt 2>&1 &
cd ..
