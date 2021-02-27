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
self_path=$(dirname $0)
export MS_SCHED_NUM=1
DEVICE_TARGET=$1
DATASET_PATH=$2
export MS_WORKER_NUM=$3
export MS_SERVER_NUM=$4
export MS_SCHED_HOST=$5
export MS_SCHED_PORT=$6

export MS_ROLE=MS_SCHED
for((i=0;i<1;i++));
do
  rm -rf ${execute_path}/sched_$i/
  mkdir ${execute_path}/sched_$i/
  cd ${execute_path}/sched_$i/ || exit
  python ${self_path}/../test_ps_embedding_heterogeneous_conv2d_adam.py --device_target=$DEVICE_TARGET --dataset_path=$DATASET_PATH &
done

export MS_ROLE=MS_PSERVER
for((i=0;i<$MS_SERVER_NUM;i++));
do
  rm -rf ${execute_path}/server_$i/
  mkdir ${execute_path}/server_$i/
  cd ${execute_path}/server_$i/ || exit
  python ${self_path}/../test_ps_embedding_heterogeneous_conv2d_adam.py --device_target=$DEVICE_TARGET --dataset_path=$DATASET_PATH &
done

export MS_ROLE=MS_WORKER
for((i=0;i<$MS_WORKER_NUM;i++));
do
  rm -rf ${execute_path}/worker_$i/
  mkdir ${execute_path}/worker_$i/
  cd ${execute_path}/worker_$i/ || exit
  python ${self_path}/../test_ps_embedding_heterogeneous_conv2d_adam.py --device_target=$DEVICE_TARGET --dataset_path=$DATASET_PATH &
done

wait $!
exit $?
