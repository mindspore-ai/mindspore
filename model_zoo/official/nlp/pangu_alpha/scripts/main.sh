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

ulimit -u unlimited
ROOT_PATH=$(cd "`dirname $0`" || exit; pwd)

# 单机多卡和分布式
if [ $# == 5 ]; then
    SERVER_NUM=$2
    RANK_SIZE=$1
    export RANK_TABLE_FILE=$3

    SERVER_ID=$4
    device_each_server=$((RANK_SIZE / SERVER_NUM))
    rank_start=$((${device_each_server} * SERVER_ID))

    # 先启动后台任务，最后留一个前台任务查看日志输出
    for((i=$(($device_each_server-1)); i>=0; i--))
    do
        rankid=$((rank_start + i))
        export DEVICE_ID=${i}
        export RANK_ID=${rankid}
        rm ${ROOT_PATH}/device$rankid/ -rf
        mkdir ${ROOT_PATH}/device$rankid
        cd ${ROOT_PATH}/device$rankid || exit
        echo "start training for rank $RANK_ID, device $DEVICE_ID"
        env > env.log

        if [ $i -eq 0 ]; then
            python ${ROOT_PATH}/train.py --distribute=true --device_num=$device_each_server --data_url=$5 --run_type=train --param_init_type=fp32 --mode=2.6B | tee log
        else
            python ${ROOT_PATH}/train.py --distribute=true --device_num=$device_each_server --data_url=$5 --run_type=train --param_init_type=fp32 --mode=2.6B &> log &
        fi
    done
else
    echo "Invalid input parameter, usage: main.sh device_count server_count RANK_TABLE_FILE server_id dataset" | tee log
    exit 1
fi

wait
