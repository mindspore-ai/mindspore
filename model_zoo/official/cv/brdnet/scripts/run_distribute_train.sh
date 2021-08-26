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

if [ $# != 8 ]; then
  echo "Usage: sh run_distribute_train.sh [train_code_path] [train_data]" \
       "[batch_size] [sigma] [channel] [epoch] [lr] [rank_table_file_path]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)/"
  fi
}

get_real_path_name() {
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

train_data=$(get_real_path $2)
echo $train_data

if [ ! -d $train_data ]
then
    echo "error: train_data=$train_data is not a dictionary."
exit 1
fi

rank_table_file_path=$(get_real_path_name $8)
echo "rank_table_file_path: "$rank_table_file_path

ulimit -c unlimited
export SLOG_PRINT_TO_STDOUT=0
export RANK_TABLE_FILE=$rank_table_file_path
export RANK_SIZE=8
export RANK_START_ID=0

for((i=0;i<=$RANK_SIZE-1;i++));
do
    export RANK_ID=${i}
    export DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='${i}', device id='${DEVICE_ID}'...'
    if [ -d ${train_code_path}device${DEVICE_ID} ]; then
      rm -rf ${train_code_path}device${DEVICE_ID}
    fi
    mkdir ${train_code_path}device${DEVICE_ID}
    cd ${train_code_path}device${DEVICE_ID} || exit
    nohup python ${train_code_path}train.py --is_distributed=1 \
    --train_data=${train_data} \
    --batch_size=$3 \
    --sigma=$4 \
    --channel=$5 \
    --epoch=$6 \
    --lr=$7 > ${train_code_path}device${DEVICE_ID}/log.txt 2>&1 &
done
