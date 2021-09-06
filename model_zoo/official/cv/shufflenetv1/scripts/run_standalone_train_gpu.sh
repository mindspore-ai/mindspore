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
# run as sh scripts/run_standalone_train.sh DEVICE_ID DATA_DIR
# limitations under the License.
# ============================================================================
get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

DEVICE_ID=$1
DATA_DIR=$(get_real_path $2)
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

BASEPATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="$BASEPATH/../gpu_default_config.yaml"
train_path=train_standalone${DEVICE_ID}

if [ -d ${train_path} ]; then
  rm -rf ${train_path}
fi
mkdir -p ${train_path}
echo "start training for device $DEVICE_ID"
cd ${train_path}|| exit

python ${BASEPATH}/../train.py  \
    --config_path=$CONFIG_FILE \
    --train_dataset_path=$DATA_DIR > log.txt 2>&1 &

