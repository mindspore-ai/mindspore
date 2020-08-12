#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train.sh [DATASET_PATH] [RESUME_YOLOV3] [MINDSPORE_HCCL_CONFIG_PATH]"
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
RESUME_YOLOV3=$(get_real_path $2)
MINDSPORE_HCCL_CONFIG_PATH=$(get_real_path $3)

echo $DATASET_PATH
echo $RESUME_YOLOV3
echo $MINDSPORE_HCCL_CONFIG_PATH

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

if [ ! -f $RESUME_YOLOV3 ]
then
    echo "error: PRETRAINED_PATH=$RESUME_YOLOV3 is not a file"
exit 1
fi

if [ ! -f $MINDSPORE_HCCL_CONFIG_PATH ]
then
    echo "error: MINDSPORE_HCCL_CONFIG_PATH=$MINDSPORE_HCCL_CONFIG_PATH is not a file"
exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export MINDSPORE_HCCL_CONFIG_PATH=$MINDSPORE_HCCL_CONFIG_PATH

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py \
        --data_dir=$DATASET_PATH \
        --resume_yolov3=$RESUME_YOLOV3 \
        --is_distributed=1 \
        --per_batch_size=16 \
        --lr=0.012 \
        --T_max=135 \
        --max_epoch=135 \
        --warmup_epochs=5 \
        --lr_scheduler=cosine_annealing  > log.txt 2>&1 &
    cd ..
done
