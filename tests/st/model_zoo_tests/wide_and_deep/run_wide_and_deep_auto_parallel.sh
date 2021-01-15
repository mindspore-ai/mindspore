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
unset SLOG_PRINT_TO_STDOUT
export MINDSPORE_HCCL_CONFIG_PATH=$CONFIG_PATH/hccl/rank_table_${DEVICE_NUM}p.json
CODE_DIR="./"
if [ -d ${BASE_PATH}/../../../../model_zoo/official/recommend/wide_and_deep ]; then
    CODE_DIR=${BASE_PATH}/../../../../model_zoo/official/recommend/wide_and_deep
elif [ -d ${BASE_PATH}/../../model_zoo/official/recommend/wide_and_deep ]; then
    CODE_DIR=${BASE_PATH}/../../model_zoo/official/recommend/wide_and_deep
else
     echo "[ERROR] code dir is not found"
fi
echo $CODE_DIR
rm -rf ${BASE_PATH}/wide_and_deep
cp -r ${CODE_DIR}  ${BASE_PATH}/wide_and_deep
cp -f ${BASE_PATH}/python_file_for_ci/train_and_test_multinpu_ci.py ${BASE_PATH}/wide_and_deep/train_and_test_multinpu_ci.py
cp -f ${BASE_PATH}/python_file_for_ci/__init__.py ${BASE_PATH}/wide_and_deep/__init__.py
cp -f ${BASE_PATH}/python_file_for_ci/config.py ${BASE_PATH}/wide_and_deep/src/config.py
cp -f ${BASE_PATH}/python_file_for_ci/callbacks.py ${BASE_PATH}/wide_and_deep/src/callbacks.py
cp -f ${BASE_PATH}/python_file_for_ci/datasets.py ${BASE_PATH}/wide_and_deep/src/datasets.py
cp -f ${BASE_PATH}/python_file_for_ci/wide_and_deep.py ${BASE_PATH}/wide_and_deep/src/wide_and_deep.py
source ${BASE_PATH}/env.sh
export PYTHONPATH=${BASE_PATH}/wide_and_deep/:$PYTHONPATH
process_pid=()
for((i=0; i<$DEVICE_NUM; i++)); do
    rm -rf ${BASE_PATH}/wide_and_deep_auto_parallel${i}
    mkdir ${BASE_PATH}/wide_and_deep_auto_parallel${i}
    cd ${BASE_PATH}/wide_and_deep_auto_parallel${i}
    export RANK_ID=${i}
    export DEVICE_ID=${i}
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v ../wide_and_deep/train_and_test_multinpu_ci.py > train_and_test_multinpu_ci$i.log 2>&1 &
    process_pid[${i}]=`echo $!`
done

for((i=0; i<${DEVICE_NUM}; i++)); do
    wait ${process_pid[i]}
    status=`echo $?`
    if [ "${status}" != "0" ]; then
        echo "[ERROR] test wide_and_deep semi auto parallel failed. status: ${status}"
        exit 1
    else
        echo "[INFO] test wide_and_deep semi auto parallel success."
    fi
done

exit 0
