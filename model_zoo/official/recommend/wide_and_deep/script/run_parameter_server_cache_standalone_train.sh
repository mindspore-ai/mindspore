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

execute_path=$(pwd)
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
export EPOCH_SIZE=$1
export DEVICE_TARGET=$2
export DATASET=$3
export MS_COMM_TYPE=zmq
export MS_SCHED_NUM=1
export MS_WORKER_NUM=1
export MS_SERVER_NUM=$4
export MS_SCHED_HOST=$5
export MS_SCHED_PORT=$6

export MS_ROLE=MS_SCHED
rm -rf ${execute_path}/sched/
mkdir ${execute_path}/sched/
cd ${execute_path}/sched/ || exit
export DEVICE_ID=$i
python -s ${self_path}/../train_and_eval_parameter_server.py --epochs=$EPOCH_SIZE --device_target=$DEVICE_TARGET --data_path=$DATASET \
                                                             --parameter_server=1 --vocab_cache_size=300000 >sched.log 2>&1 &

export MS_ROLE=MS_PSERVER
for((i=0;i<$MS_SERVER_NUM;i++));
do
  rm -rf ${execute_path}/server_$i/
  mkdir ${execute_path}/server_$i/
  cd ${execute_path}/server_$i/ || exit
  export DEVICE_ID=$i
  python -s ${self_path}/../train_and_eval_parameter_server.py --epochs=$EPOCH_SIZE --device_target=$DEVICE_TARGET --data_path=$DATASET \
                                                               --parameter_server=1 --vocab_cache_size=300000 >server_$i.log 2>&1 &
done

export MS_ROLE=MS_WORKER
rm -rf ${execute_path}/worker/
mkdir ${execute_path}/worker/
cd ${execute_path}/worker/ || exit
export DEVICE_ID=$i
python -s ${self_path}/../train_and_eval_parameter_server.py --epochs=$EPOCH_SIZE --device_target=$DEVICE_TARGET --data_path=$DATASET \
                                                             --parameter_server=1 --vocab_cache_size=300000 \
                                                             --dropout_flag=1 >worker.log 2>&1 &
