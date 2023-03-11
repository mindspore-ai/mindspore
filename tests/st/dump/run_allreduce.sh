#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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
CONFIG_PATH=/home/workspace/mindspore_config
export DEVICE_NUM=8
export RANK_SIZE=$DEVICE_NUM
unset SLOG_PRINT_TO_STDOUT
export MINDSPORE_HCCL_CONFIG_PATH=$CONFIG_PATH/hccl/rank_table_${DEVICE_NUM}p.json

process_pid=()
for((i=0; i<$DEVICE_NUM; i++)); do
    rm -rf ${BASE_PATH}/hccl_net${i}
    mkdir ${BASE_PATH}/hccl_net${i}
    cp -r ${BASE_PATH}/allreduce_net.py  ${BASE_PATH}/hccl_net${i}/
    cd ${BASE_PATH}/hccl_net${i}
    export RANK_ID=${i}
    export DEVICE_ID=${i}
    echo "start training for device $i"
    env > env$i.log
    python allreduce_net.py > test_dump_hccl_8p_log$i.log 2>&1 &
    process_pid[${i}]=`echo $!`
done

for((i=0; i<${DEVICE_NUM}; i++)); do
    wait ${process_pid[i]}
done
exit 0
