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

if [ $# != 5 ]
then 
    echo "Usage: sh run_eval_ascend.sh [IMGS_PATH] [ANNOS_PATH] [CHECKPOINT_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]"
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
PATH3=$(get_real_path $3)
PATH4=$(get_real_path $4)
echo $PATH1
echo $PATH2
echo $PATH3
echo $PATH4


if [ ! -f $PATH3 ]
then 
    echo "error: CHECKPOINT_PATH=$PATH3 is not a file"
exit 1
fi 

if [ ! -f $PATH4 ]
then 
    echo "error: COCO_TEXT_PARSER_PATH=$PATH4 is not a file"
exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export DEVICE_ID=$5
export RANK_ID=0
cp $PATH4 ../src/
if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
env > env.log
echo "start eval for device $DEVICE_ID"
python eval.py --device_id=$DEVICE_ID --imgs_path=$PATH1 --annos_path=$PATH2 --checkpoint_path=$PATH3 &> log &
cd ..
