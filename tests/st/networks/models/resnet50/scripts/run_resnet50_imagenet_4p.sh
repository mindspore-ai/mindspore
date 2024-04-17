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

BASE_PATH=$(cd "$(dirname $0)"; pwd)

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MS_WORKER_NUM=4
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118

export GLOG_v=1
export MS_ROLE=MS_SCHED
python $BASE_PATH/../train_resnet50.py >scheduler.log 2>&1 &

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ 8`
gap=`expr $avg \- 1`
rank_start=0
for((i=0; i<$MS_WORKER_NUM; i++))
do
    j=$((rank_start + i))
    start=`expr $j \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export MS_ROLE=MS_WORKER
    export MS_NODE_ID=${i}
    rm -rf $BASE_PATH/../train_parallel$j
    mkdir $BASE_PATH/../train_parallel$j
    cd $BASE_PATH/../train_parallel$j || exit
    echo "start resnet training for rank $RANK_ID, device $DEVICE_ID"
    (taskset -c $cmdopt python $BASE_PATH/../train_resnet50.py &> log; grep "#-#" log > resnet_$i.txt) &
    cd ..
done
wait
echo "result:"
cat $BASE_PATH/../train_parallel0/log
cat $BASE_PATH/../train_parallel*/resnet_*.txt
