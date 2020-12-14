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

if [ $# != 1 ]
then 
    echo "Usage: sh run_standalone_train_beta.sh [USE_DEVICE_ID]"
exit 1
fi

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}
export PYTHONPATH=${dirname_path}:$PYTHONPATH

export RANK_SIZE=1
export RANK_ID=0

USE_DEVICE_ID=$1
dev=`expr $USE_DEVICE_ID + 0`
export DEVICE_ID=$dev

EXECUTE_PATH=$(pwd)
echo *******************EXECUTE_PATH= $EXECUTE_PATH
if [ -d "${EXECUTE_PATH}/log_standalone_graph" ]; then
  echo "[INFO] Delete old data_stanalone log files"
  rm -rf ${EXECUTE_PATH}/log_standalone_graph
fi
mkdir ${EXECUTE_PATH}/log_standalone_graph


rm -rf ${EXECUTE_PATH}/data_standalone_log_$USE_DEVICE_ID
mkdir -p ${EXECUTE_PATH}/data_standalone_log_$USE_DEVICE_ID
cd ${EXECUTE_PATH}/data_standalone_log_$USE_DEVICE_ID
echo "start training for rank $RANK_ID, device $USE_DEVICE_ID"
env > ${EXECUTE_PATH}/log_standalone_graph/face_recognition_$USE_DEVICE_ID.log
python ${EXECUTE_PATH}/../train.py \
    --train_stage=beta \
    --is_distributed=0 &> ${EXECUTE_PATH}/log_standalone_graph/face_recognition_$USE_DEVICE_ID.log &

echo "[INFO] Start training..."