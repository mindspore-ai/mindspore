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
# run as sh scripts/run_eval.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
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
PATH_CHECKPOINT=$(get_real_path $3)
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASEPATH}/../gpu_default_config.yaml"

if [ -d "../eval" ]; then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

python ${BASEPATH}/../eval.py \
    --config_path=$CONFIG_FILE \
    --ckpt_path=$PATH_CHECKPOINT \
    --eval_dataset_path=$DATA_DIR > eval.log 2>&1 &
