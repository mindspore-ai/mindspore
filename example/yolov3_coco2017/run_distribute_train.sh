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

echo "Please run the scipt as: "
echo "sh run_distribute_train.sh DEVICE_NUM EPOCH_SIZE IMAGE_DIR ANNO_PATH MINDSPORE_HCCL_CONFIG_PATH"
echo "for example: sh run_distribute_train.sh 8 100 ./dataset/coco/train2017 ./dataset/train.txt　./hccl.json"
echo "After running the scipt, the network runs in the background. The log will be generated in LOGx/log.txt"

export RANK_SIZE=$1
EPOCH_SIZE=$2
IMAGE_DIR=$3
ANNO_PATH=$4
export MINDSPORE_HCCL_CONFIG_PATH=$5


for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    rm -rf LOG$i
    mkdir ./LOG$i
    cp  *.py ./LOG$i
    cd ./LOG$i || exit
    export RANK_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log
    python ../train.py  \
    --distribute=1  \
    --device_num=$RANK_SIZE  \
    --device_id=$DEVICE_ID  \
    --image_dir=$IMAGE_DIR  \
    --epoch_size=$EPOCH_SIZE  \
    --anno_path=$ANNO_PATH > log.txt 2>&1 &
    cd ../
done
