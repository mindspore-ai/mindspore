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
DATA_DIR=$1
DEVICE_ID=$2
PATH_CHECKPOINT=$3

current_exec_path=$(pwd)
echo ${current_exec_path}

curtime=`date '+%Y%m%d-%H%M%S'`

echo ${curtime} > ${current_exec_path}/eval_starttime

CUDA_VISIBLE_DEVICES=${DEVICE_ID} python ./eval.py --platform 'GPU' --data_path ${DATA_DIR} --checkpoint ${PATH_CHECKPOINT} > ${current_exec_path}/eval.log 2>&1 &
