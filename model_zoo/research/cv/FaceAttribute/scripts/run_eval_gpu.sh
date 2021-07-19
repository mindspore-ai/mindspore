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

if [ $# != 3 ]
then
    echo "Usage: sh run_eval.sh [MINDRECORD_FILE] [CUDA_VISIBLE_DEVICES] [PRETRAINED_BACKBONE]"
exit 1
fi

current_exec_path=$(pwd)
echo ${current_exec_path}

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$2
export RANK_SIZE=1

SCRIPT_NAME='eval.py'

ulimit -c unlimited
echo 'start evaluating'
export RANK_ID=0
rm -rf eval_gpu
mkdir eval_gpu
cd eval_gpu

python ${dirname_path}/${SCRIPT_NAME} \
    --mindrecord_path=$1 \
    --device_target="GPU" \
    --model_path=$3 > eval.log  2>&1 &
echo 'running'
