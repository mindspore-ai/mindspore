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

if [ $# != 7 ]; then
  echo "Usage: sh run_train.sh [train_code_path] [train_data]" \
       "[batch_size] [sigma] [channel] [epoch] [lr]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)/"
  fi
}

train_code_path=$(get_real_path $1)
echo "train_code_path: "$train_code_path

train_data=$(get_real_path $2)
echo "train_data: "$train_data

if [ ! -d $train_code_path ]
then
    echo "error: train_code_path=$train_code_path is not a dictionary."
exit 1
fi

if [ ! -d $train_data ]
then
    echo "error: train_data=$train_data is not a dictionary."
exit 1
fi

nohup python ${train_code_path}train.py --is_distributed=0 \
--train_data=${train_data} \
--batch_size=$3 \
--sigma=$4 \
--channel=$5 \
--epoch=$6 \
--lr=$7 > log.txt 2>&1 &
echo 'Train task has been started successfully!'
echo 'Please check the log at log.txt'
