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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "for example: bash run_deeplabv3_ci.sh DEVICE_ID DATA_PATH PRETRAINED_CKPT_PATH"
echo "=============================================================================================================="
DEVICE_ID=$1
DATA_DIR=$2
PATH_CHECKPOINT=$3
BASE_PATH=$(cd "$(dirname $0)"; pwd)
unset SLOG_PRINT_TO_STDOUT
CODE_DIR="./"
if [ -d ${BASE_PATH}/../../../../tests/models/deeplabv3 ]; then
    CODE_DIR=${BASE_PATH}/../../../../tests/models/deeplabv3
elif [ -d ${BASE_PATH}/../../tests/models/deeplabv3 ]; then
    CODE_DIR=${BASE_PATH}/../../tests/models/deeplabv3
else
     echo "[ERROR] code dir is not found"
fi
echo $CODE_DIR
rm -rf ${BASE_PATH}/deeplabv3
cp -r ${CODE_DIR}  ${BASE_PATH}/deeplabv3
cp -f ${BASE_PATH}/train_one_epoch_with_loss.py ${BASE_PATH}/deeplabv3/train_one_epoch_with_loss.py
cd ${BASE_PATH}/deeplabv3
python train_one_epoch_with_loss.py --data_url=$DATA_DIR --checkpoint_url=$PATH_CHECKPOINT --device_id=$DEVICE_ID > train_deeplabv3_ci.log 2>&1 &
process_pid=`echo $!`
wait ${process_pid}
status=`echo $?`
if [ "${status}" != "0" ]; then
        echo "[ERROR] test deeplabv3 failed. status: ${status}"
    exit 1
else
    echo "[INFO] test deeplabv3 success."
fi