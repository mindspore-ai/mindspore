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

current_exec_path=$(pwd)
echo ${current_exec_path}

curtime=`date '+%Y%m%d-%H%M%S'`
RANK_SIZE=8

rm ${current_exec_path}/device_parallel/ -rf
mkdir ${current_exec_path}/device_parallel
echo ${curtime} > ${current_exec_path}/device_parallel/starttime

mpirun --allow-run-as-root -n $RANK_SIZE python ${current_exec_path}/train.py \
                                                --GPU \
                                                --distributed \
                                                --data_path ${DATA_DIR} \
                                                --cur_time ${curtime} > ${current_exec_path}/device_parallel/efficientnet_b0.log 2>&1 &
