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

if [ $# != 3 ]; then
  echo "Usage: sh run_distribute_train.sh [train_code_path][RANK_TABLE_FILE][DATA_PATH]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

train_code_path=$(get_real_path $1)
echo $train_code_path

if [ ! -d $train_code_path ]
then
    echo "error: train_code_path=$train_code_path is not a dictionary."
exit 1
fi

RANK_TABLE_FILE=$(get_real_path $2)
echo $RANK_TABLE_FILE

if [ ! -f $RANK_TABLE_FILE ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file."
exit 1
fi

DATA_PATH=$(get_real_path $3)
echo $DATA_PATH

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a dictionary."
exit 1
fi

ulimit -c unlimited
export SLOG_PRINT_TO_STDOUT=0
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export RANK_SIZE=8
export RANK_START_ID=0


for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=${i}
    export DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='${i}', device id='${DEVICE_ID}'...'
    if [ -d ${train_code_path}/device${DEVICE_ID} ]; then
      rm -rf ${train_code_path}/device${DEVICE_ID}
    fi
    mkdir ${train_code_path}/device${DEVICE_ID}
    cd ${train_code_path}/device${DEVICE_ID} || exit
    python ${train_code_path}/deep_sort/deep/train.py    --data_url=${DATA_PATH}   \
                                               --train_url=./checkpoint   \
                                               --run_distribute=True   \
                                               --run_modelarts=False > out.log 2>&1 &
done
