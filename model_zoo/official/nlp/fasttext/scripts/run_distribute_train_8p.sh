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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_distributed_train.sh DATASET_PATH RANK_TABLE_PATH"
echo "for example: sh run_distributed_train.sh /home/workspace/ag /home/workspace/rank_table_file.json"
echo "It is better to use absolute path."
echo "=============================================================================================================="
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET=$(get_real_path $1)
echo $DATASET
DATANAME=$(basename $DATASET)
RANK_TABLE_PATH=$(get_real_path $2)
echo $DATANAME
if [ ! -d $DATASET ]
then
  echo "Error: DATA_PATH=$DATASET is not a file"
exit 1
fi
current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_TABLE_FILE=$RANK_TABLE_PATH


echo $RANK_TABLE_FILE
export RANK_SIZE=8
export DEVICE_NUM=8


for((i=0;i<=7;i++));
do
    rm -rf ${current_exec_path}/device$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    cp ../../*.py ./
    cp -r ../../src ./
    cp -r ../*.sh ./
    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    python ../../train.py --data_path $DATASET --data_name $DATANAME > log_fasttext.log 2>&1 &
    cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit
