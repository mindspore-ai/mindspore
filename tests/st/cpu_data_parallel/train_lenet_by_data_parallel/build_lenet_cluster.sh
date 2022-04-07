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
export MS_SCHED_PORT=8117

# Launch 1 scheduler.
export MS_ROLE=MS_SCHED
python3 train_lenet_by_data_parallel.py >scheduler.txt 2>&1 &
echo "scheduler start success!"

# Launch 8 workers.
export MS_ROLE=MS_WORKER
process_pid=()
for((i=0;i<${MS_WORKER_NUM};i++));
do
    python3 train_lenet_by_data_parallel.py >worker_$i.txt 2>&1 &
    echo "worker ${i} start success with pid ${!}"
    process_pid[${i}]=${!}
done

# Check the execution result of each node.
for((i=0; i<${MS_WORKER_NUM}; i++));
do
    wait ${process_pid[${i}]}
    status=${?}
    if [ ${status} != 0 ]; then
        echo "[ERROR] train lenet on worker $i failed. status: ${status}"
        exit 1
    else
        echo "[INFO] train lenet on worker $i success."
    fi
done

exit 0
