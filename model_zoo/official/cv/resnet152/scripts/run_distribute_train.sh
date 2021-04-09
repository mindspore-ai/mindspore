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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train.sh RANK_TABLE_FILE DATA_PATH PRETRAINED_CKPT_PATH](optional)"
echo "For example: bash run_distribute_train.sh hccl_8p_01234567_127.0.0.1.json /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ $# == 3 ]
then 
    PATH3=$(get_real_path $5)
fi

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -d $PATH2 ]
then 
    echo "error: DATA_PATH=$PATH2 is not a directory"
exit 1
fi 

if [ $# == 3 ] && [ ! -f $PATH3 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH3 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

DATA_PATH=$2
export DATA_PATH=${DATA_PATH}

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ../*.py ./device$i
    cp *.sh ./device$i
    cp -r ../src ./device$i
    cd ./device$i || exit
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log

    if [ $# == 2 ]
    then
        python train.py --run_distribute=True  --data_url=$PATH2 &> train.log &
    fi
    
    if [ $# == 3 ]
    then
        python train.py --run_distribute=True  --data_url=$PATH2 --pre_trained=$PATH3 &> train.log &
    fi

    cd ../
done
