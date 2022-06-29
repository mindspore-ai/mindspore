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
export MS_WORKER_NUM=1
export MS_SERVER_NUM=1
export MS_SCHED_HOST=$2
export MS_SCHED_PORT=$3

export MS_ROLE=MS_SCHED
rm -rf ${self_path}/sched/
mkdir ${self_path}/sched/
cd ${self_path}/sched/ || exit
python ${self_path}/test_embedding_cache_standalone.py --device_target=$DEVICE_TARGET >sched.log 2>&1 &
sched_pid=`echo $!`

export MS_ROLE=MS_PSERVER
rm -rf ${self_path}/server/
mkdir ${self_path}/server/
cd ${self_path}/server/ || exit
python ${self_path}/test_embedding_cache_standalone.py --device_target=$DEVICE_TARGET >server.log 2>&1 &
server_pid=`echo $!`

export MS_ROLE=MS_WORKER
rm -rf ${self_path}/worker/
mkdir ${self_path}/worker/
cd ${self_path}/worker/ || exit
export RANK_ID=0
python ${self_path}/test_embedding_cache_standalone.py --device_target=$DEVICE_TARGET &>worker.log 2>&1 &
worker_pid=`echo $!`

wait ${worker_pid}
status=`echo $?`
if [ "${status}" != "0" ]; then
    echo "[ERROR] test_embedding_cache_standalone failed, wait worker failed, status: ${status}"
    exit 1
fi

wait ${server_pid}
status=`echo $?`
if [ "${status}" != "0" ]; then
    echo "[ERROR] test_embedding_cache_standalone failed, wait server failed, status: ${status}"
    exit 1
fi

wait ${sched_pid}
status=`echo $?`
if [ "${status}" != "0" ]; then
    echo "[ERROR] test_embedding_cache_standalone failed, wait scheduler failed, status: ${status}"
    exit 1
fi
