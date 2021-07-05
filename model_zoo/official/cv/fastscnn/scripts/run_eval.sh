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

if [ $# != 5 ]; then
  echo "Usage: sh run_eval.sh [train_code_path] [dataset] [resume_path]" \
       "[resume_name] [output_path]"
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

dataset=$(get_real_path $2)
echo "dataset: "$dataset

resume_path=$(get_real_path $3)
echo "resume_path: "$resume_path

if [ ! -d $train_code_path ]
then
    echo "error: train_code_path=$train_code_path is not a dictionary."
exit 1
fi

if [ ! -d $dataset ]
then
    echo "error: dataset=$dataset is not a dictionary."
exit 1
fi

if [ ! -d $resume_path ]
then
    echo "error: resume_path=$resume_path is not a dictionary."
exit 1
fi

python ${train_code_path}eval.py --is_distributed=0 \
--dataset=$dataset \
--resume_path=$resume_path \
--resume_name=$4 \
--output_path=$5 > log.txt 2>&1 & 
