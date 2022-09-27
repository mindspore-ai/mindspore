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

export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=$2
export GLOG_v=1

# Launch 1 scheduler.
export MS_ROLE=MS_SCHED
python3 $1 >scheduler.log 2>&1 &
sched_pid=${!}
echo "scheduler start success!"

# Launch 8 workers.
export MS_ROLE=MS_WORKER
process_pid=()
for((i=0;i<8;i++));
do
    python3 $1 >worker_$i.log 2>&1 &
    echo "worker ${i} start success with pid ${!}"
    process_pid[${i}]=${!}
done

wait ${sched_pid}

# Check the execution result of each node.
for((i=0; i<${MS_WORKER_NUM}; i++));
do
    wait ${process_pid[${i}]}
    status=${?}
    if [ ${status} != 0 ]; then
        echo "[ERROR] run all reduce on worker $i failed. status: ${status}"
        exit 1
    else
        echo "[INFO] run all reduce on worker $i success."
    fi
done

exit 0
