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
# ==========================================================================

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
PATH4=$4
PATH5=$5
echo "========================================"
echo "$PATH1"
echo "$PATH2"
echo "$PATH3"
echo "$PATH4"
echo "$PATH5"
echo "========================================"

if [ ! -d $PATH2 ]
then
    echo "error: DATA_URL=$PATH2 is not a directory"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1
export MINDSPORE_HCCL_CONFIG_PATH=$PATH1

DATA_URL=$2
export DATA_URL=${DATA_URL}

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp *.py ./device$i
    cp -r scripts ./device$i
    cp -r src ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log

    if [ $# == 5 ]
    then
        python train.py \
          --data_path=$PATH2 \
          --pretrained_ckpt=$PATH3 \
          --log_dir=$PATH4 \
          --modelart=$PATH5 &> train.log &
    fi

    cd ../
done
