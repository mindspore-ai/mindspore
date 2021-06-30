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

if [ $# != 1 ]
then
    echo "Usage: bash run_standalone_train_gpu.sh [CONFIG_PATH]"
    exit 1
fi

current_exec_path=$(pwd)
echo ${current_exec_path}

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}

export PYTHONPATH=${dirname_path}:$PYTHONPATH

export RANK_SIZE=1

SCRIPT_NAME='train.py'

ulimit -c unlimited

CONFIG_PATH=$1

export RANK_ID=0
rm -rf ${current_exec_path}/train
mkdir ${current_exec_path}/train
cd ${current_exec_path}/train || exit

python ${dirname_path}/${SCRIPT_NAME} --config_path=$CONFIG_PATH > train.log  2>&1 &
