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

if [ $# != 1 ]
then 
    echo "Usage: sh run_distribute_train_base.sh [RANK_TABLE_FILE]"
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

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

# Distribute config 
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

EXECUTE_PATH=$(pwd)
echo *******************EXECUTE_PATH=$EXECUTE_PATH
if [ -d "${EXECUTE_PATH}/log_parallel_graph" ]; then
  echo "[INFO] Delete old data_parallel log files"
  rm -rf ${EXECUTE_PATH}/log_parallel_graph
fi
mkdir ${EXECUTE_PATH}/log_parallel_graph

for((i=0;i<=7;i++));
do
    rm -rf ${EXECUTE_PATH}/data_parallel_log_$i
    mkdir -p ${EXECUTE_PATH}/data_parallel_log_$i
    cd ${EXECUTE_PATH}/data_parallel_log_$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > ${EXECUTE_PATH}/log_parallel_graph/face_recognition_$i.log
    python ${EXECUTE_PATH}/../train.py \
    --train_stage=base \
    --is_distributed=1 &> ${EXECUTE_PATH}/log_parallel_graph/face_recognition_$i.log &
done
echo "[INFO] Start training..."