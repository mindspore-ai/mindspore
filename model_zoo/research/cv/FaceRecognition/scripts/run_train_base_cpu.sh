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

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}
export PYTHONPATH=${dirname_path}:$PYTHONPATH


USE_DEVICE_ID=0
dev=`expr $USE_DEVICE_ID + 0`
export DEVICE_ID=$dev

EXECUTE_PATH=$(pwd)
echo *******************EXECUTE_PATH= $EXECUTE_PATH
if [ -d "${EXECUTE_PATH}/log_standalone_graph" ]; then
  echo "[INFO] Delete old data_standalone log files"
  rm -rf ${EXECUTE_PATH}/log_standalone_graph
fi
mkdir ${EXECUTE_PATH}/log_standalone_graph


rm -rf ${EXECUTE_PATH}/data_standalone_log_$USE_DEVICE_ID
mkdir -p ${EXECUTE_PATH}/data_standalone_log_$USE_DEVICE_ID
cd ${EXECUTE_PATH}/data_standalone_log_$USE_DEVICE_ID || exit

env > ${EXECUTE_PATH}/log_standalone_graph/face_recognition_$USE_DEVICE_ID.log
python ${EXECUTE_PATH}/../train.py \
    --config_path=${EXECUTE_PATH}/../base_config_cpu.yaml \
    --train_stage=base \
    --is_distributed=0 &> ${EXECUTE_PATH}/log_standalone_graph/face_recognition_$USE_DEVICE_ID.log &

echo "[INFO] Start training..."