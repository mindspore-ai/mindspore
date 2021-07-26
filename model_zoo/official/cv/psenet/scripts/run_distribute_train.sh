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

current_exec_path=$(pwd)
echo 'current_exec_path: '${current_exec_path}

if [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train.sh [RANK_FILE] [PRETRAINED_PATH] [TRAIN_ROOT_DIR]"
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
if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi

if [ ! -z "${2}" ];
then
  PATH2=$(get_real_path $2)
else
  PATH2=$2
fi

PATH3=$(get_real_path $3)
if [ ! -d $PATH3 ]
then
    echo "error: TRAIN_ROOT_DIR=$PATH3 is not a directory"
exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

for((i=0; i<${DEVICE_NUM}; i++))
do
    if [ -d ${current_exec_path}/device_$i/ ]
    then
        if [ -d ${current_exec_path}/device_$i/checkpoints/ ]
        then
            rm ${current_exec_path}/device_$i/checkpoints/ -rf
        fi

        if [ -f ${current_exec_path}/device_$i/loss.log ]
        then
            rm ${current_exec_path}/device_$i/loss.log
        fi

        if [ -f ${current_exec_path}/device_$i/test_deep$i.log ]
        then
            rm ${current_exec_path}/device_$i/test_deep$i.log
        fi
    else
        mkdir ${current_exec_path}/device_$i
    fi

    cd ${current_exec_path}/device_$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python ${current_exec_path}/train.py --run_distribute=True --pre_trained=$PATH2 --TRAIN_ROOT_DIR=$PATH3 >test_deep$i.log 2>&1 &
    cd ${current_exec_path} || exit
done
