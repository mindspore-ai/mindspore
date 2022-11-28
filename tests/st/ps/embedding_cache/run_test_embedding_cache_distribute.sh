#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
DEVICE_TARGET=$1
export RANK_SIZE=$2
export MS_WORKER_NUM=$RANK_SIZE
export MS_SERVER_NUM=2
export MS_SCHED_HOST=$3
export MS_SCHED_PORT=$4
export SPARSE=$5

CONFIG_PATH=/home/workspace/mindspore_config
export MINDSPORE_HCCL_CONFIG_PATH=$CONFIG_PATH/hccl/rank_tabel_4p/rank_table_${RANK_SIZE}p_1.json

if [[ ! -n "$5" ]]; then
  export SPARSE=0
fi

export MS_ROLE=MS_SCHED
rm -rf ${self_path}/sched/
mkdir ${self_path}/sched/
cd ${self_path}/sched/ || exit
python ${self_path}/test_embedding_cache_distribute.py --device_target=$DEVICE_TARGET >sched.log 2>&1 &
sched_pid=`echo $!`

export MS_ROLE=MS_PSERVER
server_pids=()
for((i=0;i<$MS_SERVER_NUM;i++));
do
  rm -rf ${self_path}/server_$i/
  mkdir ${self_path}/server_$i/
  cd ${self_path}/server_$i/ || exit
  python ${self_path}/test_embedding_cache_distribute.py --device_target=$DEVICE_TARGET --sparse=$SPARSE >server_$i.log 2>&1 &
  server_pids[${i}]=`echo $!`
done

export MS_ROLE=MS_WORKER
worker_pids=()
for((i=0;i<$MS_WORKER_NUM;i++));
do
  rm -rf ${self_path}/worker_$i/
  mkdir ${self_path}/worker_$i/
  cd ${self_path}/worker_$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  python ${self_path}/test_embedding_cache_distribute.py --device_target=$DEVICE_TARGET --sparse=$SPARSE &>worker_$i.log 2>&1 &
  worker_pids[${i}]=`echo $!`
done

for((i=0; i<${MS_WORKER_NUM}; i++)); do
    wait ${worker_pids[i]}
    status=`echo $?`
    if [ "${status}" != "0" ]; then
        echo "[ERROR] test_embedding_cache_distribute failed, wait worker_${i} failed, status: ${status}"
        exit 1
    fi
done

for((i=0; i<${MS_SERVER_NUM}; i++)); do
    wait ${server_pids[i]}
    status=`echo $?`
    if [ "${status}" != "0" ]; then
        echo "[ERROR] test_embedding_cache_distribute failed, wait server_${i} failed, status: ${status}"
        exit 1
    fi
done

wait ${sched_pid}
status=`echo $?`
if [ "${status}" != "0" ]; then
    echo "[ERROR] test_embedding_cache_distribute failed, wait scheduler failed, status: ${status}"
    exit 1
fi
