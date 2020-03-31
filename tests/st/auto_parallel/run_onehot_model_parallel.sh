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
set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
CONFIG_PATH=/home/workspace/mindspore_config
export DEVICE_NUM=8
export RANK_SIZE=$DEVICE_NUM
ulimit -n 65535
source ${BASE_PATH}/env.sh
unset SLOG_PRINT_TO_STDOUT
export MINDSPORE_HCCL_CONFIG_PATH=$CONFIG_PATH/hccl/rank_table_${DEVICE_NUM}p.json

process_pid=()
for((i=0; i<$DEVICE_NUM; i++)); do
    rm -rf ${BASE_PATH}/onehot_model_parallel${i}
    mkdir ${BASE_PATH}/onehot_model_parallel${i}
    cp -r ${BASE_PATH}/onehot_model_parallel.py  ${BASE_PATH}/onehot_model_parallel${i}/
    cd ${BASE_PATH}/onehot_model_parallel${i}
    export RANK_ID=${i}
    export DEVICE_ID=${i}
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v onehot_model_parallel.py > onehot_model_parallel_log$i.log 2>&1 &
    process_pid[${i}]=`echo $!`
done

for((i=0; i<${DEVICE_NUM}; i++)); do
    wait ${process_pid[i]}
    status=`echo $?`
    if [ "${status}" != "0" ]; then
        echo "[ERROR] test_onehot_model_parallel failed. status: ${status}"
        exit 1
    else
        echo "[INFO] test_onehot_model_parallel success."
    fi
done

exit 0
