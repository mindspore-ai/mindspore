#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
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
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
export ENABLE_CELL_REUSE=1  # lazyinline
export MS_MEMORY_STATISTIC=1  # memory statistic
source ${BASE_PATH}/env.sh
unset SLOG_PRINT_TO_STDOUT
export RANK_TABLE_FILE=$CONFIG_PATH/hccl/rank_table_${DEVICE_NUM}p.json

yaml_name=$1
DATASET_PATH=$2
CHECK_POINT_PATH=$3
YAML_FILE=$BASE_PATH/configs/$yaml_name

export START_DEVICE_ID=0


rm -rf ${BASE_PATH}/llama_finetune*
process_pid=()
for((i=0; i<$DEVICE_NUM; i++)); do
    mkdir ${BASE_PATH}/llama_finetune${i}
    cp -r ${BASE_PATH}/train_llama.py  ${BASE_PATH}/llama_finetune${i}/
        cp -r ${BASE_PATH}/mindformers ${BASE_PATH}/llama_finetune${i}/
    cd ${BASE_PATH}/llama_finetune${i}
    export RANK_ID=${i}
    export DEVICE_ID=$((i + START_DEVICE_ID))
    echo "start llama training for rank_id $i, device_id ${DEVICE_ID}"
    env > env$i.log
    python train_llama.py --checkpoint_path $CHECK_POINT_PATH --yaml_file $YAML_FILE --dataset_path $DATASET_PATH \
    --use_parallel True --batch_size 1 --rank_id $RANK_ID --device_num $DEVICE_NUM > finetune_llama_log$i.log 2>&1 &
    process_pid[${i}]=`echo $!`
done

for((i=0; i<${DEVICE_NUM}; i++)); do
    wait ${process_pid[i]}
    status=`echo $?`
    if [ "${status}" != "0" ]; then
        echo "[ERROR] test_train_llama failed. status: ${status}"
        exit 1
    else
        echo "[INFO] test_train_llama success."
    fi
done

exit 0
