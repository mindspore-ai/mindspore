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

#if [ $# != 2 ]
#then
#    echo "Usage: sh run_standalone_train.sh [DATASET_PATH] [TRAIN_IMG_DIR] [TRAIN_JSON_FILE] [VAL_IMG_DIR] [VAL_JSON_FILE]"
#exit 1
#fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)
TRAIN_IMG_DIR=$2
TRAIN_JSON_FILE=$3
VAL_IMG_DIR=$4
VAL_JSON_FILE=$5

echo $DATASET_PATH

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp -r ../src ./train
cp -r ../model_utils ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log

python train.py \
    --data_dir=$DATASET_PATH \
    --train_img_dir=$TRAIN_IMG_DIR \
    --train_json_file=$TRAIN_JSON_FILE \
    --val_img_dir=$VAL_IMG_DIR \
    --val_json_file=$VAL_JSON_FILE \
    --is_distributed=0 \
    --lr=0.01 \
    --t_max=500 \
    --max_epoch=500 \
    --warmup_epochs=20 \
    --training_shape=416 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
cd ..