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
BASE_PATH=$(dirname "$(dirname "$(readlink -f $0)")")
dataset_type='MR'
CONFIG_FILE="${BASE_PATH}/mr_config.yaml"
if [ $# == 1 ]
then
    if [ $1 == "MR" ]; then
        CONFIG_FILE="${BASE_PATH}/mr_config.yaml"
    elif [ $1 == "SUBJ" ]; then
        CONFIG_FILE="${BASE_PATH}/subj_config.yaml"
    elif [ $1 == "SST2" ]; then
        CONFIG_FILE="${BASE_PATH}/sst2_config.yaml"
    else
        echo "error: the selected dataset is not in supported set{MR, SUBJ, SST2}"
    exit 1
    fi
    dataset_type=$1
fi
rm ./ckpt_0 -rf
python train.py --dataset=$dataset_type --config_path=$CONFIG_FILE --output_path './output' > train.log 2>&1 &
