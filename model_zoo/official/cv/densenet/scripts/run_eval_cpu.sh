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

if [ $# -lt 4 ]
then
    echo "Usage: sh run_eval_cpu.sh [NET_NAME] [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH]"
    exit 1
fi

# check checkpoint file
if [ ! -f $4 ]
then
    echo "error: CHECKPOINT_PATH=$4 is not a file"
    exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

python ${BASEPATH}/../eval.py \
            --net=$1 \
            --dataset=$2 \
            --data_dir=$3 \
            --device_target='CPU' \
            --is_distributed=0 \
            --pretrained=$4 > eval.log 2>&1 &
