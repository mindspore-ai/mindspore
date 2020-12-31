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

if [ $# -ne 5 ]
then 
    echo "Usage: sh run_distribute_train_ascend.sh [IMGS_PATH] [ANNOS_PATH] [RANK_TABLE_FILE] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH]"
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
echo $PATH1
PATH2=$(get_real_path $2)
echo $PATH2
PATH3=$(get_real_path $3)
echo $PATH3
PATH4=$(get_real_path $4)
echo $PATH4
PATH5=$(get_real_path $5)
echo $PATH5

if [ ! -f $PATH3 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH3 is not a file"
exit 1
fi 

if [ ! -f $PATH4 ]
then 
    echo "error: PRETRAINED_PATH=$PATH4 is not a file"
exit 1
fi

if [ ! -f $PATH5 ]
then 
    echo "error: COCO_TEXT_PARSER_PATH=$PATH5 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH3
cp $PATH5 ../src/

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py --device_id=$i --rank_id=$i --imgs_path=$PATH1 --annos_path=$PATH2 --run_distribute=True --device_num=$DEVICE_NUM --pre_trained=$PATH4 &> log &
    cd ..
done
