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

CURPATH="$(dirname "$0")"
# shellcheck source=/dev/null
. ${CURPATH}/cache_util.sh

if [ $# != 3 ] && [ $# != 4 ] && [ $# != 5 ]
then
  echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)"
  echo "       bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)"
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
PATH2=$(get_real_path $2)
CONFIG_FILE=$(get_real_path $3)

if [ $# == 4 ]
then 
    PATH3=$(get_real_path $4)
fi

if [ $# == 5 ]
then
  RUN_EVAL=$4
  EVAL_DATASET_PATH=$(get_real_path $5)
fi

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -d $PATH2 ]
then 
    echo "error: DATASET_PATH=$PATH2 is not a directory"
exit 1
fi 

if [ $# == 4 ] && [ ! -f $PATH3 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH3 is not a file"
exit 1
fi

if [ "x${RUN_EVAL}" == "xTrue" ] && [ ! -d $EVAL_DATASET_PATH ]
then
  echo "error: EVAL_DATASET_PATH=$EVAL_DATASET_PATH is not a directory"
  exit 1
fi

if [ "x${RUN_EVAL}" == "xTrue" ]
then
  bootup_cache_server
  CACHE_SESSION_ID=$(generate_cache_session)
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $DEVICE_NUM`
gap=`expr $avg \- 1`

for((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../config/*.yaml ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    if [ $# == 3 ]
    then    
        taskset -c $cmdopt python train.py --run_distribute=True --device_num=$RANK_SIZE --data_path=$PATH2 \
        --config_path=$CONFIG_FILE --output_path './output' &> log &
    fi
    
    if [ $# == 4 ]
    then
        taskset -c $cmdopt python train.py --run_distribute=True --device_num=$RANK_SIZE --data_path=$PATH2 --pre_trained=$PATH3 \
        --config_path=$CONFIG_FILE --output_path './output' &> log &
    fi

    if [ $# == 5 ]
    then
      taskset -c $cmdopt python train.py --run_distribute=True --device_num=$RANK_SIZE --data_path=$PATH2 \
      --run_eval=$RUN_EVAL --eval_dataset_path=$EVAL_DATASET_PATH --enable_cache=True \
      --cache_session_id=$CACHE_SESSION_ID --config_path=$CONFIG_FILE --output_path './output' &> log &
      if [ "x${RUN_EVAL}" == "xTrue" ]
      then
        echo -e "\nWhen training run is done, remember to shut down the cache server via \"cache_admin --stop\""
      fi
    fi
    cd ..
done
