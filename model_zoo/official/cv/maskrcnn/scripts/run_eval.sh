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

if [ $# != 3 ]
then 
    echo "Usage: bash run_eval.sh [ANN_FILE] [CHECKPOINT_PATH] [DATA_PATH]"
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
echo $PATH1
echo $PATH2
echo $PATH3

if [ ! -f $PATH1 ]
then 
    echo "error: ANN_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -f $PATH2 ]
then 
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export DEVICE_ID=0
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp ../*.yaml ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
env > env.log_eval
echo "start eval for device $DEVICE_ID"
python ./eval.py --device_id=$DEVICE_ID --ann_file=$PATH1 --checkpoint_path=$PATH2 \
--coco_root=$PATH3 &> log_eval.txt &
cd ..
