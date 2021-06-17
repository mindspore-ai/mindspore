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

if [ $# != 1 ] && [ $# != 2 ]
then 
    echo "run as scripts/run_distribute_train_ascend.sh RANK_TABLE_FILE PRED_TRAINED(options)"
exit 1
fi

current_exec_path=$(pwd)
echo ${current_exec_path}

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
echo $PATH1

PATH2=$(get_real_path $2)
echo $PATH2

export RANK_TABLE_FILE=$PATH1
export RANK_SIZE=8
ulimit -u unlimited
for((i=0;i<$RANK_SIZE;i++));
do
    rm ./train_parallel_$i/ -rf
    mkdir ./train_parallel_$i
    cp ./*.py ./train_parallel_$i
    cp ./scripts/*.sh ./train_parallel_$i
    cp -r ./src ./train_parallel_$i
    cp ./*yaml ./train_parallel_$i
    cd ./train_parallel_$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    if [ -f $PATH2 ]
    then
      python train.py --PRED_TRAINED=$PATH2 --run_distribute=True >log_$i.log 2>&1 &
    else
      python train.py --run_distribute=True >log_$i.log 2>&1 &
    fi
    cd .. || exit
done

